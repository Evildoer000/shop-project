from app.services.organizer_dataset import (
    ACTIVE_DATASET_SUBDIRS,
    load_organizer_products,
    resolve_dataset_dir,
)


def test_loads_organizer_dataset() -> None:
    products = load_organizer_products()
    dataset_dir = resolve_dataset_dir()
    expected_count = sum(
        len(list((dataset_dir / subdir / "data").glob("*.json")))
        for subdir in ACTIVE_DATASET_SUBDIRS
    )

    assert expected_count > 0
    assert len(products) == expected_count
    assert products[0]["product_id"].startswith("p_")
    assert products[0]["structured_attributes"]["source"] == "organizer_dataset"
    assert "source_payload" not in products[0]["structured_attributes"]
    assert products[0]["stock"] == 1
    assert products[0]["sales"] is None
