import json

DEFAULT_INPUT_XLSX_FILE="./dummy.xlsx"
DEFAULT_OUTPUT_JSONL_FILE="./output2.jsonl"

class ExcelToJSONLParser:
    def __init__(self, input_file, output_file, config):
        self.input_file = input_file
        self.output_file = output_file
        self.config = config

        # 缓冲区：用于存放当前正在收集的大步骤
        self.current_major_record = None

    def _read_rows_from_excel(self):
        """
        【终极断网版】纯 Python 标准库 Excel 读取器
        完全不需要 openpyxl，依赖自带的 zipfile 和 xml
        """
        import zipfile
        import xml.etree.ElementTree as ET
        import re

        # 辅助函数：把 Excel 的字母列号(A, B, C, AA) 转换为数字索引(0, 1, 2, 26)
        def col_to_index(col_str):
            num = 0
            for c in col_str:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
            return num - 1

        with zipfile.ZipFile(self.input_file, 'r') as z:
            # 第一步：解析 sharedStrings.xml 获取所有的文本字典
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                with z.open('xl/sharedStrings.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    # Python 3.8+ 支持 {*}` 通配符忽略命名空间
                    for si in root.findall('.//{*}si'):
                        # 尝试获取 t 标签中的文本
                        t_elem = si.find('.//{*}t')
                        if t_elem is not None and t_elem.text is not None:
                            shared_strings.append(t_elem.text)
                        else:
                            shared_strings.append("")

            # 第二步：解析第一个工作表 (通常是 xl/worksheets/sheet1.xml)
            # 在极少数复杂文件中名字可能不同，但 99% 的简单表格都是这个路径
            sheet_path = 'xl/worksheets/sheet1.xml'
            if sheet_path not in z.namelist():
                raise FileNotFoundError(f"在 XLSX 包内找不到 {sheet_path}！")

            with z.open(sheet_path) as f:
                tree = ET.parse(f)
                root = tree.getroot()

                # 遍历所有的行 <row>
                for i, row in enumerate(root.findall('.//{*}row')):
                    # 跳过表头配置
                    if i < self.config.get("skip_rows", 0):
                        continue

                    row_data = []
                    # 遍历行内的每一个单元格 <c>
                    for c in row.findall('.//{*}c'):
                        # 获取单元格的坐标，例如 "A1", "B2", "AA3"
                        r_attr = c.attrib.get('r', '')
                        # 提取出字母部分 (列名)
                        col_letters = re.sub(r'[0-9]', '', r_attr)
                        col_idx = col_to_index(col_letters) if col_letters else len(row_data)

                        # 补齐因 Excel 省略空白单元格导致的列表长度不足
                        while len(row_data) <= col_idx:
                            row_data.append(None)

                        # 获取单元格的值 <v>
                        v_elem = c.find('.//{*}v')
                        if v_elem is not None and v_elem.text is not None:
                            val = v_elem.text
                            # 检查单元格类型属性 t="s" 代表它是一个共享字符串索引
                            if c.attrib.get('t') == 's':
                                val = shared_strings[int(val)]
                            row_data[col_idx] = val

                    # 将这一行的列表 yield 出去给 parse 主循环处理
                    yield row_data

    def parse(self):
        # 决定使用真实 Excel 还是模拟数据
        row_generator = self._read_rows_from_excel()

        with open(self.output_file, 'w', encoding='utf-8') as out_f:

            for row_index, row in enumerate(row_generator):
                # 过滤掉完全为空的空行
                if not any(cell is not None and str(cell).strip() != "" for cell in row):
                    continue

                # 【解耦点1】：判断这一行是大步骤(a)还是小步骤(b)
                if self.config["is_major_step"](row):
                    # 遇到新的 a 时，如果缓冲区里已经有上一个 a 的数据，就先把它写进 JSONL
                    if self.current_major_record is not None:
                        list_key = self.config["list_key"]
                        self.current_major_record[list_key] = smalls
                        out_f.write(json.dumps(self.current_major_record, ensure_ascii=False) + '\n')

                    # 清空并初始化一个新的大步骤 a
                    self.current_major_record = {}
                    smalls = []

                    # 【解耦点2】：根据大步骤的配置，提取 key-value
                    for json_key, col_index in self.config["major_mapping"].items():
                        # 防止越界保护
                        val = row[col_index] if col_index < len(row) else None
                        # 【修改这里】：强制将 None 转换为空字符串 ""
                        self.current_major_record[json_key] = val if val is not None else ""

                else:
                    # 这是一个小步骤 b
                    if self.current_major_record is None:
                        # 容错处理：如果一开始就遇到了 b，但前面还没出现过 a，直接跳过或者记录日志
                        print(f"Warning: Row {row_index} is a minor step but no major step found yet.")
                        continue

                    minor_obj = {}
                    # 【解耦点3】：根据小步骤的配置，提取 key-value
                    for json_key, col_index in self.config["minor_mapping"].items():
                        val = row[col_index] if col_index < len(row) else None

                        # 可选：如果提取到的值是空的，你要不要记录进 json 里？这里我们全部保留。
                        # 可以把 None 转成空字符串以保持格式统一
                        minor_obj[json_key] = val if val is not None else ""

                    # 把这个小步骤丢进当前大步骤的列表中
                    smalls.append(minor_obj)

            # 循环结束后，别忘了把最后一个大步骤写入文件
            if self.current_major_record is not None:
                list_key = self.config["list_key"]
                self.current_major_record[list_key] = smalls
                out_f.write(json.dumps(self.current_major_record, ensure_ascii=False) + '\n')


# ==========================================
# 下面是使用示例和解耦配置 (你的核心修改区)
# ==========================================
if __name__ == "__main__":

    import argparse
    cli_parser = argparse.ArgumentParser(description="excel .xlsx to .jsonl")
    cli_parser.add_argument("-i", "--input", required=False, default=DEFAULT_INPUT_XLSX_FILE, help="input excel file path")
    cli_parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUTPUT_JSONL_FILE, help="output jsonl file path")
    args = cli_parser.parse_args()

    # 假设你的 Excel 列结构是：
    # 列0: 大步骤序号 | 列1: 大步骤名称 | 列2: 注意事项 | 列3: 小步骤操作类型 | 列4: 小步骤操作

    excel_config = {
        # 跳过第一行表头
        "skip_rows": 1,

        # 存放小步骤列表的 key 名称
        "list_key": "步骤列表",

        # 【核心逻辑】：如何判断这一行是 a (大步骤)？
        # 这里的逻辑是：如果第 1 列 (即 B 列 “说明”) 有值，说明它是大步骤。
        "is_major_step": lambda row: row[1] is not None and str(row[0]).strip() != "",

        # 大步骤 (a) 需要提取哪些列？ -> { "你的 JSON Key" : Excel 列索引(从0开始) }
        "major_mapping": {
            "用例名称": 1,  # 提取第 1 列
            "注意事项": 2  # 提取第 2 列
        },

        # 小步骤 (b) 需要提取哪些列？
        "minor_mapping": {
            "注意事项": 2,
            "操作类型": 3,
            "操作": 4,
            "操作对象": 5,
            "操作目的": 6,
            "判据": 7,
            "描述": 8,
            "左值": 9,
            "右值": 10,
            "单位": 11,
            "是否使用设备": 12,
            "设备指令数量": 13,
            "设备单元号": 14,
            "设备指令号": 15,
        }
    }

    input_file = args.input
    output_file = args.output

    # 测试运行 (use_mock=True 代表使用捏造的内部数据，不用真正去读 Excel)
    parser = ExcelToJSONLParser(input_file, output_file, excel_config)
    parser.parse()

    print("解析完成！请查看 output_excel.jsonl 文件。")