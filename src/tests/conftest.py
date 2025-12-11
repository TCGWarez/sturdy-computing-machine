"""
src/tests/conftest.py: Pytest configuration and shared fixtures
Following PRD.md Task 12 specifications

Provides:
- Common fixtures for testing
- Test database setup/teardown
- Temporary file management
- Logging configuration
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging

import numpy as np
from PIL import Image

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope='session')
def test_data_dir():
    """
    Session-scoped fixture for test data directory

    Returns:
        Path to test data directory
    """
    base_dir = Path(__file__).parent.parent.parent
    test_data = base_dir / 'data' / 'test_data'
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


@pytest.fixture
def temp_dir():
    """
    Function-scoped temporary directory

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path

    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_card_image():
    """
    Create a sample card image for testing

    Returns:
        PIL Image object
    """
    # Create 512x512 card image
    img = Image.new('RGB', (512, 512), color='white')

    # Add some structure to make it look like a card
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    # Card border
    draw.rectangle([10, 10, 502, 502], outline='black', width=3)

    # Name region (top 15%)
    draw.rectangle([20, 20, 492, 96], fill='lightgray', outline='black')

    # Art region (middle)
    draw.rectangle([20, 106, 492, 350], fill='lightblue', outline='black')

    # Text box (bottom)
    draw.rectangle([20, 360, 492, 480], fill='lightyellow', outline='black')

    # Collector number region (bottom-left)
    draw.text((30, 490), "001", fill='black')

    return img


@pytest.fixture
def sample_card_image_file(temp_dir, sample_card_image):
    """
    Save sample card image to temporary file

    Args:
        temp_dir: Temporary directory fixture
        sample_card_image: Sample card image fixture

    Returns:
        Path to saved image file
    """
    img_path = temp_dir / "sample_card.jpg"
    sample_card_image.save(img_path)
    return img_path


@pytest.fixture
def sample_embedding():
    """
    Create a sample embedding vector

    Returns:
        numpy array of shape (512,) with random values
    """
    np.random.seed(42)
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def mock_card_data():
    """
    Mock card data for testing

    Returns:
        Dict with card information
    """
    return {
        'id': 'test_card_001',
        'scryfall_id': 'abc123def456',
        'name': 'Test Card',
        'set_code': 'TST',
        'collector_number': '001',
        'finish': 'nonfoil',
        'image_path': '/path/to/test_card.jpg'
    }


@pytest.fixture
def mock_card_list():
    """
    Mock list of cards for testing

    Returns:
        List of card data dicts
    """
    cards = []
    card_names = ['Lightning Bolt', 'Giant Growth', 'Counterspell', 'Llanowar Elves', 'Serra Angel']

    for i, name in enumerate(card_names):
        cards.append({
            'id': f'test_card_{i:03d}',
            'scryfall_id': f'scryfall_{i:03d}',
            'name': name,
            'set_code': 'TST',
            'collector_number': f'{i+1:03d}',
            'finish': 'nonfoil',
            'image_path': f'/path/to/{name.lower().replace(" ", "_")}.jpg'
        })

    return cards


# Database fixtures

@pytest.fixture
def test_db(temp_dir):
    """
    Create temporary test database

    Yields:
        SQLAlchemy session for test database
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.schema import Base

    # Create temporary database
    db_path = temp_dir / 'test.db'
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSession()

    yield session

    # Cleanup
    session.close()
    engine.dispose()


@pytest.fixture
def test_db_with_cards(test_db, mock_card_list):
    """
    Test database pre-populated with mock cards

    Args:
        test_db: Test database session
        mock_card_list: List of mock card data

    Yields:
        Database session with cards
    """
    from src.database.schema import Card, PhashVariant, CompositeEmbedding
    from src.indexing.indexer import serialize_embedding

    # Insert cards
    for card_data in mock_card_list:
        card = Card(**card_data)
        test_db.add(card)

        # Add pHash variants
        for variant_type in ['full', 'name', 'collector']:
            phash = PhashVariant(
                card_id=card_data['id'],
                variant_type=variant_type,
                phash=123456 + mock_card_list.index(card_data)
            )
            test_db.add(phash)

        # Add composite embedding
        embedding_vector = np.random.randn(512).astype(np.float32)
        embedding = CompositeEmbedding(
            card_id=card_data['id'],
            embedding=serialize_embedding(embedding_vector)
        )
        test_db.add(embedding)

    test_db.commit()

    yield test_db


# Pytest hooks

def pytest_configure(config):
    """
    Pytest configuration hook

    Args:
        config: Pytest config object
    """
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (require database, indexes)")
    config.addinivalue_line("markers", "slow: Slow tests (> 1 second)")
    config.addinivalue_line("markers", "fixtures: Tests requiring fixture dataset")


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection

    Args:
        config: Pytest config object
        items: List of collected test items
    """
    # Auto-mark tests based on naming conventions
    for item in items:
        # Mark integration tests
        if 'integration' in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if 'slow' in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)


# Fixtures for specific test modules

@pytest.fixture
def rotated_card_image():
    """
    Create a card image rotated by known angle for detector tests

    Returns:
        Tuple of (image, rotation_angle)
    """
    import cv2

    # Create base card image
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Draw white rectangle (card)
    card_width, card_height = 400, 280
    center_x, center_y = 500, 500

    # Define corners
    corners = np.array([
        [-card_width//2, -card_height//2],
        [card_width//2, -card_height//2],
        [card_width//2, card_height//2],
        [-card_width//2, card_height//2]
    ], dtype=np.float32)

    # Rotate 45 degrees
    angle = 45
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_corners = corners @ rotation_matrix.T
    rotated_corners += [center_x, center_y]

    # Draw rotated card
    cv2.fillPoly(img, [rotated_corners.astype(np.int32)], (255, 255, 255))

    return img, angle, rotated_corners
