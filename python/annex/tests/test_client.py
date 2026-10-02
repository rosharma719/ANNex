from annex import AnnexError


def test_annex_error_stores_status_and_message():
    err = AnnexError(404, "not found")
    assert err.status_code == 404
    assert err.message == "not found"
    assert "404" in str(err)
    assert "not found" in str(err)
