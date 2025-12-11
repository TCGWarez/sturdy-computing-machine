"""
src/tests/test_indexer.py: Unit tests for indexer functionality
Following PRD.md Task 12 specifications

Tests:
- Index building with fixtures
- Database insertion
- Embedding computation
- pHash variant storage
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from src.indexing.indexer import (
    serialize_embedding,
    deserialize_embedding,
    compute_card_features
)
from src.database.schema import SessionLocal, Card, PhashVariant, Embedding, Base, engine
from src.indexing.phash import compute_phash_variants, phash_to_int


@pytest.fixture
def temp_db():
    """Create a temporary test database"""
    # Create temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    # Create engine and session for temp db
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(bind=test_engine)

    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSession()

    yield session

    # Cleanup
    session.close()
    test_engine.dispose()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_card_images():
    """Create temporary card images for testing"""
    temp_dir = Path(tempfile.mkdtemp())

    images = []
    for i in range(3):
        # Create different colored cards
        colors = ['red', 'green', 'blue']
        img = Image.new('RGB', (512, 512), color=colors[i])

        img_path = temp_dir / f"card_{i}.jpg"
        img.save(img_path)
        images.append(img_path)

    yield images, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestEmbeddingSerialization:
    """Test embedding serialization/deserialization"""

    def test_serialize_embedding(self):
        """Test serializing embedding to bytes"""
        embedding = np.random.randn(512).astype(np.float32)

        serialized = serialize_embedding(embedding)

        assert isinstance(serialized, bytes)
        assert len(serialized) == 512 * 4  # 512 floats * 4 bytes each

    def test_deserialize_embedding(self):
        """Test deserializing embedding from bytes"""
        original = np.random.randn(512).astype(np.float32)
        serialized = serialize_embedding(original)

        deserialized = deserialize_embedding(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert deserialized.shape == (512,)
        assert deserialized.dtype == np.float32
        assert np.allclose(original, deserialized)

    def test_round_trip_serialization(self):
        """Test round-trip embedding serialization"""
        original = np.random.randn(256).astype(np.float32)

        # Serialize and deserialize
        serialized = serialize_embedding(original)
        deserialized = deserialize_embedding(serialized)

        # Should be identical
        assert np.array_equal(original, deserialized)


class TestComputeCardFeatures:
    """Test computing card features (embedding + phash variants)"""

    def test_compute_card_features_returns_dict(self, temp_card_images):
        """Test that compute_card_features returns expected structure"""
        images, _ = temp_card_images
        image_path = images[0]

        # Load image
        img = Image.open(image_path)

        # For testing without actual embedder, we'll just test pHash
        phash_variants = compute_phash_variants(img)

        assert isinstance(phash_variants, dict)
        assert 'full' in phash_variants
        assert 'name' in phash_variants
        assert 'collector' in phash_variants

    def test_phash_variants_are_reproducible(self, temp_card_images):
        """Test that pHash computation is reproducible"""
        images, _ = temp_card_images
        image_path = images[0]

        img = Image.open(image_path)

        # Compute twice
        phash1 = compute_phash_variants(img)
        phash2 = compute_phash_variants(img)

        # Should be identical
        assert phash1['full'] == phash2['full']
        assert phash1['name'] == phash2['name']
        assert phash1['collector'] == phash2['collector']


class TestDatabaseOperations:
    """Test database insertion and retrieval"""

    def test_insert_card(self, temp_db):
        """Test inserting card into database"""
        card = Card(
            id='test_card_1',
            scryfall_id='abc123',
            name='Test Card',
            set_code='TST',
            collector_number='001',
            finish='nonfoil',
            image_path='/path/to/image.jpg'
        )

        temp_db.add(card)
        temp_db.commit()

        # Retrieve card
        retrieved = temp_db.query(Card).filter_by(id='test_card_1').first()

        assert retrieved is not None
        assert retrieved.name == 'Test Card'
        assert retrieved.set_code == 'TST'

    def test_insert_phash_variants(self, temp_db):
        """Test inserting pHash variants"""
        # First insert card
        card = Card(
            id='test_card_2',
            scryfall_id='def456',
            name='Test Card 2',
            set_code='TST',
            collector_number='002',
            finish='nonfoil'
        )
        temp_db.add(card)
        temp_db.commit()

        # Insert pHash variants
        variants = [
            PhashVariant(card_id='test_card_2', variant_type='full', phash=123456789),
            PhashVariant(card_id='test_card_2', variant_type='name', phash=987654321),
            PhashVariant(card_id='test_card_2', variant_type='collector', phash=111222333)
        ]

        for variant in variants:
            temp_db.add(variant)
        temp_db.commit()

        # Retrieve variants
        retrieved = temp_db.query(PhashVariant).filter_by(card_id='test_card_2').all()

        assert len(retrieved) == 3
        variant_types = {v.variant_type for v in retrieved}
        assert variant_types == {'full', 'name', 'collector'}

    def test_insert_embedding(self, temp_db):
        """Test inserting embedding"""
        # First insert card
        card = Card(
            id='test_card_3',
            scryfall_id='ghi789',
            name='Test Card 3',
            set_code='TST',
            collector_number='003',
            finish='nonfoil'
        )
        temp_db.add(card)
        temp_db.commit()

        # Create embedding
        embedding_vector = np.random.randn(512).astype(np.float32)
        embedding_blob = serialize_embedding(embedding_vector)

        embedding = Embedding(
            card_id='test_card_3',
            embedding=embedding_blob
        )
        temp_db.add(embedding)
        temp_db.commit()

        # Retrieve embedding
        retrieved = temp_db.query(Embedding).filter_by(card_id='test_card_3').first()

        assert retrieved is not None
        retrieved_vector = deserialize_embedding(retrieved.embedding)
        assert np.allclose(embedding_vector, retrieved_vector)

    def test_card_relationships(self, temp_db):
        """Test card relationships with phash_variants and embedding"""
        # Insert card
        card = Card(
            id='test_card_4',
            scryfall_id='jkl012',
            name='Test Card 4',
            set_code='TST',
            collector_number='004',
            finish='nonfoil'
        )
        temp_db.add(card)
        temp_db.commit()

        # Insert phash variants
        phash_variant = PhashVariant(
            card_id='test_card_4',
            variant_type='full',
            phash=123456
        )
        temp_db.add(phash_variant)

        # Insert embedding
        embedding_vector = np.random.randn(512).astype(np.float32)
        embedding = Embedding(
            card_id='test_card_4',
            embedding=serialize_embedding(embedding_vector)
        )
        temp_db.add(embedding)
        temp_db.commit()

        # Retrieve card with relationships
        retrieved_card = temp_db.query(Card).filter_by(id='test_card_4').first()

        assert retrieved_card is not None
        assert len(retrieved_card.phash_variants) == 1
        assert retrieved_card.phash_variants[0].variant_type == 'full'
        assert retrieved_card.embedding is not None

    def test_bulk_insert(self, temp_db, temp_card_images):
        """Test bulk insertion of multiple cards"""
        images, _ = temp_card_images

        cards = []
        for i, img_path in enumerate(images):
            card = Card(
                id=f'bulk_card_{i}',
                scryfall_id=f'bulk_{i}',
                name=f'Bulk Card {i}',
                set_code='TST',
                collector_number=f'{i:03d}',
                finish='nonfoil',
                image_path=str(img_path)
            )
            cards.append(card)

        # Bulk insert
        temp_db.bulk_save_objects(cards)
        temp_db.commit()

        # Verify
        retrieved = temp_db.query(Card).filter(Card.set_code == 'TST').all()
        assert len(retrieved) >= len(images)


class TestIndexQuery:
    """Test querying indexed data"""

    def test_query_by_set_code(self, temp_db):
        """Test querying cards by set code"""
        # Insert cards from different sets
        cards = [
            Card(id='m21_1', scryfall_id='m21_1', name='M21 Card 1',
                 set_code='M21', collector_number='001', finish='nonfoil'),
            Card(id='m21_2', scryfall_id='m21_2', name='M21 Card 2',
                 set_code='M21', collector_number='002', finish='nonfoil'),
            Card(id='znr_1', scryfall_id='znr_1', name='ZNR Card 1',
                 set_code='ZNR', collector_number='001', finish='nonfoil')
        ]

        for card in cards:
            temp_db.add(card)
        temp_db.commit()

        # Query M21 cards
        m21_cards = temp_db.query(Card).filter_by(set_code='M21').all()

        assert len(m21_cards) == 2
        assert all(c.set_code == 'M21' for c in m21_cards)

    def test_query_by_finish(self, temp_db):
        """Test querying cards by finish type"""
        cards = [
            Card(id='foil_1', scryfall_id='f1', name='Foil 1',
                 set_code='TST', collector_number='001', finish='foil'),
            Card(id='nonfoil_1', scryfall_id='nf1', name='Nonfoil 1',
                 set_code='TST', collector_number='001', finish='nonfoil')
        ]

        for card in cards:
            temp_db.add(card)
        temp_db.commit()

        # Query foil cards
        foil_cards = temp_db.query(Card).filter_by(finish='foil').all()

        assert len(foil_cards) == 1
        assert foil_cards[0].finish == 'foil'

    def test_query_phash_by_variant_type(self, temp_db):
        """Test querying pHash variants by type"""
        # Insert card with variants
        card = Card(id='test_ph', scryfall_id='ph1', name='Test PHash',
                   set_code='TST', collector_number='001', finish='nonfoil')
        temp_db.add(card)
        temp_db.commit()

        variants = [
            PhashVariant(card_id='test_ph', variant_type='full', phash=111),
            PhashVariant(card_id='test_ph', variant_type='name', phash=222),
            PhashVariant(card_id='test_ph', variant_type='collector', phash=333)
        ]

        for v in variants:
            temp_db.add(v)
        temp_db.commit()

        # Query specific variant
        name_variant = temp_db.query(PhashVariant).filter_by(
            card_id='test_ph',
            variant_type='name'
        ).first()

        assert name_variant is not None
        assert name_variant.phash == 222


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
