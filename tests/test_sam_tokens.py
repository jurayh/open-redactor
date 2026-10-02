from open_redactor.sam_client import parse_output_text


def test_parse_live_token_box():
    text = "<0f>0<|box;x1=10;y1=20;x2=30;y2=50;w=100;h=80|><|mask;x=0;y=0;data=abc|>"
    result = parse_output_text(text, phrase="face", shape=(80, 100))
    assert len(result.objects) == 1
    assert result.objects[0].box == (10, 20, 30, 50)
    assert result.tracks["0"][0].sum() == 20 * 30
