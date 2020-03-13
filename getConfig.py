import configparser
def get_config(config_file='config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')
    # 获取整型参数、按照key-value的形式保存
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    # 获取浮点型参数、按照key-value的形式保存
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    # 获取字符型参数，按照key-value的形式保存
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    # 返回一个字典对象， 包含所读取的参数
    return dict(_conf_ints + _conf_floats + _conf_strings)
