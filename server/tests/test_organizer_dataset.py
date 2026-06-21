from app.services.organizer_dataset import load_organizer_products


def test_loads_organizer_dataset() -> None:
    products = load_organizer_products()

    assert len(products) == 100
    assert products[0]["product_id"].startswith("p_")
    assert products[0]["structured_attributes"]["source"] == "organizer_dataset"
    assert "source_payload" not in products[0]["structured_attributes"]
    assert products[0]["stock"] is None
    assert products[0]["sales"] is None

