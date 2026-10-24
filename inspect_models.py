
import joblib
from pathlib import Path

for path in Path("models").glob("*.pkl"):
    print(f"\n{'=' * 50}")
    print("File:", path.name)

    try:
        obj = joblib.load(path)

        print("Object type:", type(obj).__name__)

        if hasattr(obj, "classes_"):
            print("Classes:", obj.classes_)

        if hasattr(obj, "named_steps"):
            print("Pipeline steps:", list(obj.named_steps.keys()))

        if hasattr(obj, "n_features_in_"):
            print("Expected input features:", obj.n_features_in_)

        if hasattr(obj, "vocabulary_"):
            print("Vocabulary size:", len(obj.vocabulary_))

        if hasattr(obj, "get_params"):
            print("Parameters:", obj.get_params())

    except Exception as e:
        print("Could not load:", repr(e))
