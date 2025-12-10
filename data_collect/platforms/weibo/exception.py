class WeiboError(Exception):
    """基础异常"""

class CookieInvalidError(WeiboError):
    """Cookie 失效或被要求登录"""

class ApiError(WeiboError):
    """调用 m.weibo.cn API 失败"""

class ParseError(WeiboError):
    """解析数据失败"""