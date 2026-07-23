import xml.etree.ElementTree as ET
import json
import re
import html

DEFAULT_INPUT_XML_FILE = r"./dummy2.xml"
DEFAULT_OUTPUT_JSONL_FILE = r"./output.jsonl"

class XMLToJSONLParser:
    def __init__(self, xml_file, output_file, config):
        self.xml_file = xml_file
        self.output_file = output_file
        self.config = config

        # 需求2：用一个列表模拟栈，维护当前到达的光标路径
        self.tag_stack = []

        # 需求3：用一个字典维护经过标签的属性 (Attributes)。
        # 为了防止不同层级的同名属性冲突（例如外层有number，内层也有number），
        # 我们使用 "完整路径" 作为字典的 Key，值是该标签的属性字典。
        self.attr_state = {}

        # 当前正在收集的这一行的数据（一行代表 JSONL 中的一个 JSON Object）
        self.current_record = {}

    def _get_current_path(self):
        """返回当前的绝对路径，例如: Root/Orders/RouteOperation"""
        return "/".join(self.tag_stack)

    def clean_html_text(self, raw_text):
        """
        更新需求：网页文本清洗器（提取图片、处理各类占位符、转义字符、网页换行）
        """
        if not raw_text or not raw_text.strip():
            return ""

        # 1. 反转义 XML 实体
        # 这一步非常强大，它会自动把 &#xa0;, &nbsp;, &lt;, &gt; 等全部还原为真实字符
        # 例如 &#xa0; 会被转换为真实的 \xa0 (不换行空格)
        text = html.unescape(raw_text)

        # 2. 【核心新增】提取图片并转换为 AI 友好的特殊 Token
        # 应对你提到的没有尖括号的奇葩格式: img alt="" ... src="/path.jpg"
        # 正则解释：
        # <?img\b   -> 匹配可选的 < 以及 img 单词
        # [^>]*?    -> 匹配 src 前面的任意属性 (如 style, alt 等)，非贪婪模式
        # src=["\']([^"\']+)["\'] -> 核心！精准捕获 src="路径" 里的路径内容
        # [^>]*>?   -> 匹配剩余的属性并吃掉结尾可能的 >
        def replace_img(match):
            src_url = match.group(1)
            # 转换为对 AI 友好的特殊 Token 结构
            return f"[IMAGE: {src_url}]"

        text = re.sub(r'<?img\b[^>]*?src=["\']([^"\']+)["\'][^>]*>?', replace_img, text, flags=re.IGNORECASE)

        # 3. 替换常见的 HTML 换行和空格标签
        text = re.sub(r'(?i)<br\s*/?>', '\n', text)
        text = re.sub(r'(?i)</?p[^>]*>', '\n', text)
        text = re.sub(r'(?i)<tab\s*/?>', '\t', text)

        # 4. 【核心新增】彻底过滤不可见占位符
        # html.unescape 已经把 &#xa0; 变成了 \xa0。我们将它和 &nbsp; 统一替换为普通空格
        # text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
        text = text.replace('\xa0', '').replace('&nbsp;', ' ')
        # 顺手干掉网页编辑器常生成的“零宽空格”等恶心占位符
        text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)

        # 5. 扒掉其他不关心的 HTML 标签
        # (因为图片已经被换成了 [IMAGE: xxx] 是方括号，所以这里剥离尖括号 <> 不会误伤图片！)
        text = re.sub(r'<[^>]+>', '', text)

        # 6. 清理多余的连续空白字符（把连续三个以上换行压缩为两个等）
        text = re.sub(r'\n\s*\n', '\n', text)

        return text.strip()

    def _match_path(self, current_path, target_paths):
        """
        核心匹配引擎：支持绝对路径和以 */ 开头的相对路径匹配
        target_paths: 可以是字典的 keys()，也可以是单个字符串
        """
        if not target_paths:
            return None

        # 兼容单字符串传入 (比如 row_trigger_path)
        if isinstance(target_paths, str):
            target_paths = [target_paths]

        # 优先查绝对路径精确匹配 (速度最快)
        if current_path in target_paths:
            return current_path

        # 再查模糊匹配 (处理 */layer3/layer4)
        for target_path in target_paths:
            if target_path.startswith("*/"):
                # 把 "*/layer3/layer4" 变成 "/layer3/layer4"
                suffix = target_path[1:]
                # 只要当前绝对路径是以该后缀结尾的，即算命中
                if current_path.endswith(suffix):
                    return target_path
        return None

    def parse(self):
        with open(self.output_file, 'w', encoding='utf-8') as out_f:

            # 使用 iterparse 进行流式遍历 (需求1)
            # events=('start', 'end') 表示在进入标签和离开标签时都会触发事件
            context = ET.iterparse(self.xml_file, events=('start', 'end'))

            for event, elem in context:
                # 很多 XML 标签带有命名空间，例如 {http://xxx}RouteOperation
                # 这一步把命名空间截掉，只保留干净的标签名
                tag_name = elem.tag.split('}')[-1]

                if event == 'start':
                    # 【进入标签】：压入栈
                    self.tag_stack.append(tag_name)
                    current_path = self._get_current_path()

                    # 记录该标签身上的属性 (Attributes)
                    if elem.attrib:
                        # 以绝对路径为 key 保存属性，完美解决同名属性冲突
                        self.attr_state[current_path] = elem.attrib.copy()

                        # 【升级点1】使用 match_path 动态捕获属性
                        matched_attr_path = self._match_path(current_path, self.config.get('target_attrs', {}))
                        if matched_attr_path:
                            for attr_name, output_key in self.config['target_attrs'][matched_attr_path].items():
                                if attr_name in elem.attrib:
                                    self.current_record[output_key] = elem.attrib[attr_name]

                elif event == 'end':
                    current_path = self._get_current_path()

                    # 【升级点2】使用 match_path 动态捕获文本
                    matched_text_path = self._match_path(current_path, self.config.get('target_texts', {}))
                    if matched_text_path:
                        output_key = self.config['target_texts'][matched_text_path]
                        cleaned_text = self.clean_html_text(elem.text)
                        self.current_record[output_key] = cleaned_text

                    # 【升级点3】使用 match_path 识别行的闭合点
                    if self._match_path(current_path, self.config.get('row_trigger_path')):
                        if self.current_record:
                            out_f.write(json.dumps(self.current_record, ensure_ascii=False) + '\n')
                            self.current_record = {}

                    # 【离开标签】：清理内存并弹出栈
                    elem.clear() # 极大地节省内存
                    if current_path in self.attr_state:
                        del self.attr_state[current_path]
                    self.tag_stack.pop()

# ==========================================
# 下面是使用示例和解耦配置 (你可以随意修改这里)
# ==========================================
if __name__ == "__main__":

    import argparse
    cli_parser = argparse.ArgumentParser(description="excel .xlsx to .jsonl")
    cli_parser.add_argument("-i", "--input", required=False, default=DEFAULT_INPUT_XML_FILE, help="input excel file path")
    cli_parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUTPUT_JSONL_FILE, help="output jsonl file path")
    args = cli_parser.parse_args()

    # ---------------------------------------------------------
    # 【解耦配置字典】：你的核心修改区
    # 你不需要改上面的解析代码，只需要修改这个字典来映射你的表格
    # ---------------------------------------------------------
    extraction_config = {
        # 只要经过任意层级最终到达了 RouteOperation，都算作一行数据的闭合！
        "row_trigger_path": {
            "*/RouteOperation",
            "*/WorkDecription",
        },

        # 想要从哪些路径的标签中，提取特定的属性(Attributes)？
        "target_attrs": {
            # 当到达这个路径时，抓取其 number 和 name 属性
            # 捕获任何嵌套深度下的 SubTask 的 number 属性
            "*/RouteOperation": {
                "gbName": "项目名称",
            },
            # 如果你有两个不同的标签叫 name，你依然可以写绝对路径避免冲突
            "Root/Factory/RouteOperation": {
            },
        },

        # 想要提取哪些标签里的主要文本(Text)？
        "target_texts": {
            # 前面是路径，后面是你的 JSON 表头
            # 只要是以 /OperationContext 结尾的标签，无论外面套了多少层，通通抓走！
            "*/OperationContext": "项目说明"
        }
    }

    input_file = args.input
    output_file = args.output

    # 运行解析器
    parser = XMLToJSONLParser(input_file, output_file, extraction_config)
    parser.parse()

    print("解析完成！请查看 output.jsonl 文件。")