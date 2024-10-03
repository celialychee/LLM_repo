
def check_str_useful(content: str):
    useful_char_cnt = 0
    # 有效字符范围为英文、数字、中日韩字符集
    scopes = [['a', 'z'], ['\u4e00', '\u9fff'], ['A', 'Z'], ['0', '9']]
    for char in content:
        for scope in scopes:
            if char >= scope[0] and char <= scope[1]:
                useful_char_cnt += 1
                break
    if useful_char_cnt / len(content) <= 0.5:
        # Garbled characters
        return False
    return True