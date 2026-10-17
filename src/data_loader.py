"""
data_loader.py — Dataset Loading & Category Mapping
=====================================================
Loads the 20 Newsgroups dataset (built into scikit-learn) and maps
its 20 fine-grained newsgroup labels into 6 human-friendly categories.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np

# ── Category Mapping ────────────────────────────────────────────────
# Maps each of the 20 original newsgroup names → one of 6 clean labels.
CATEGORY_MAP = {
    "comp.graphics": "Technology",
    "comp.os.ms-windows.misc": "Technology",
    "comp.sys.ibm.pc.hardware": "Technology",
    "comp.sys.mac.hardware": "Technology",
    "comp.windows.x": "Technology",
    "rec.autos": "Recreation",
    "rec.motorcycles": "Recreation",
    "rec.sport.baseball": "Sports",
    "rec.sport.hockey": "Sports",
    "sci.crypt": "Science",
    "sci.electronics": "Science",
    "sci.med": "Science",
    "sci.space": "Science",
    "misc.forsale": "Business",
    "talk.politics.guns": "Politics",
    "talk.politics.mideast": "Politics",
    "talk.politics.misc": "Politics",
}

# Only keep newsgroups that we've mapped (skip religion / atheism groups)
SELECTED_NEWSGROUPS = list(CATEGORY_MAP.keys())

# Ordered list of our 6 target categories
CATEGORIES = sorted(set(CATEGORY_MAP.values()))

# Emoji labels for the UI
CATEGORY_EMOJI = {
    "Business": "🏢",
    "Politics": "⚡",
    "Recreation": "🚗",
    "Science": "🔬",
    "Sports": "🏀",
    "Technology": "💻",
}


def load_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load the 20 Newsgroups dataset and return it with mapped labels.

    Returns
    -------
    X_train : list[str]   — training article texts
    X_test  : list[str]   — testing article texts
    y_train : np.ndarray   — training labels (mapped category names)
    y_test  : np.ndarray   — testing labels  (mapped category names)
    """
    # Fetch the subset of newsgroups we care about
    dataset = fetch_20newsgroups(
        subset="all",
        categories=SELECTED_NEWSGROUPS,
        shuffle=True,
        random_state=random_state,
        remove=("headers", "footers", "quotes"),  # remove metadata noise
    )

    texts = dataset.data
    # Map numeric labels → newsgroup name → our category
    labels = np.array(
        [CATEGORY_MAP[dataset.target_names[label]] for label in dataset.target]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return X_train, X_test, y_train, y_test


def get_sample_articles() -> dict[str, str]:
    """Return hand-written sample articles for the demo UI."""
    return {
        "💻 Technology": (
            "Apple has unveiled its latest M4 chip, promising a 50% improvement "
            "in CPU performance and dramatically better GPU capabilities for "
            "machine learning workloads. The new processor will power the next "
            "generation of MacBook Pro laptops and is built on a cutting-edge "
            "3-nanometer process. Developers are excited about the Neural Engine "
            "upgrades that enable on-device AI inference at unprecedented speeds."
        ),
        "🏀 Sports": (
            "In a thrilling overtime finish, the Lakers defeated the Celtics 118-115 "
            "in Game 7 of the NBA Finals. LeBron James scored 38 points and grabbed "
            "12 rebounds to lead his team to the championship. The victory marks the "
            "franchise's 18th title, tying them with the Celtics for the most in "
            "league history. The MVP trophy was awarded to James for the fifth time."
        ),
        "🔬 Science": (
            "NASA's James Webb Space Telescope has captured the most detailed images "
            "ever taken of a newly forming star system 1,300 light-years from Earth. "
            "The infrared observations reveal complex organic molecules in the "
            "protoplanetary disk, suggesting the building blocks of life may be "
            "common in our galaxy. Researchers published their findings in Nature."
        ),
        "⚡ Politics": (
            "The United Nations General Assembly voted overwhelmingly to approve a "
            "new climate resolution requiring all member nations to cut carbon "
            "emissions by 45% before 2035. The resolution, backed by over 150 "
            "countries, also establishes a $100 billion annual fund to help "
            "developing nations transition to renewable energy sources."
        ),
        "🏢 Business": (
            "Amazon announced record quarterly revenue of $158 billion, driven by "
            "strong growth in its cloud computing division AWS and a surge in "
            "online retail sales. The company also revealed plans to hire 50,000 "
            "additional workers and invest $10 billion in new fulfillment centers "
            "across the Midwest. Shares rose 8% in after-hours trading."
        ),
        "🚗 Recreation": (
            "The 2025 Harley-Davidson Road Glide has been completely redesigned with "
            "a new Milwaukee-Eight 121 engine producing 115 horsepower. Riders are "
            "praising the improved suspension and the new touchscreen infotainment "
            "system. The motorcycle market continues to grow as more people embrace "
            "two-wheeled touring for weekend getaways and cross-country road trips."
        ),
    }
