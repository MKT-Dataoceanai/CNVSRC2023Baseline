from transforms import TextTransform
text_transform = TextTransform()

def text2idstr(text):
    token_id_str = " ".join(
        map(str, [_.item() for _ in text_transform.tokenize(text)])
    )
    return token_id_str