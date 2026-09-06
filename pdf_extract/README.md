# 矢量 PDF 表格提取流程

`pdf_extract` 是一个可独立复制和运行的最小流程，用于把含有矢量表格线、文本和内嵌图片的 PDF 转换为：

1. 便于人工检查的结构化 HTML；
2. 保留坐标信息的长版 JSON；
3. 适合输入生成模型的短版 JSON。

流程依赖 PDF 中可读取的文字和矢量线条。纯扫描图片 PDF 应先经过 OCR 或版面识别，不属于本流程的直接输入范围。

## 目录

```text
pdf_extract/
├── run_pipeline.py              # 唯一需要直接运行的入口
├── requirements.txt
├── README.md
├── src/
│   ├── pdf_to_html.py           # PDF 几何、文字和图片 -> 中间 HTML
│   ├── structured.py            # 表格语义、嵌套、图文顺序和输出精简
│   └── pipeline.py              # 三阶段流程编排
└── example/
    ├── sample.pdf               # 流程的最初输入文件
    └── output/                  # 已成功生成的示例结果
        ├── document.cells.html
        ├── document.structured.html
        ├── document.structured.long.json
        ├── document.structured.short.json
        └── images/
```

`example/sample.pdf` 是整个示例流程的最初输入文件。由于拿到的原始 AO 范本是图片型 PDF，这个样例是我们参照该图片 PDF 手工重绘得到的矢量 PDF；同时额外嵌入了一张图片，用于验证图片导出、上下文定位和本地超链接功能。它用于验证本流程，不代表程序能把任意图片型 PDF 自动转换成矢量 PDF。

## 安装

建议使用 Python 3.10 或 3.11：

```bash
python -m pip install -r requirements.txt
```

离线环境可以先在联网机器下载依赖：

```bash
python -m pip download -r requirements.txt -d wheelhouse
```

把整个目录和 `wheelhouse/` 复制到离线服务器后安装：

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

## 运行

在 `pdf_extract` 目录执行：

```bash
python run_pipeline.py example/sample.pdf \
  --output-dir run_output \
  --headers "工种,序号,工序内容|序号,项目内容"
```

Windows PowerShell 可以写成一行：

```powershell
python run_pipeline.py example\sample.pdf --output-dir run_output --headers "工种,序号,工序内容|序号,项目内容"
```

## 配置目标表头

`--headers` 用来指定需要提取的表格：

- 同一个表格的列名用英文逗号 `,` 分隔；
- 多种可接受的表头用竖线 `|` 分隔；
- 表头文字应与 PDF 中显示的列名对应，程序会进行空白符等基础归一化。

例如：

```text
工种,序号,工序内容|序号,项目内容|序号,检查操作程序,响应及显示
```

表示同时识别：

```text
工种 / 序号 / 工序内容
序号 / 项目内容
序号 / 检查操作程序 / 响应及显示
```

如果事先不知道一个表格应当横向读取还是纵向读取，并且也没有提供预先设计好的表头信息，仅靠方框和文字坐标无法可靠确定哪一行或哪一列是表头，因此无法保证得到正确的结构化内容。当前程序默认按纵向表格读取：第一行为表头，后续内容按列向下读取；这种默认形式已在示例流程中验证可用。其他阅读方向应先提供相应表头规则，或扩展结构判定逻辑。

识别到的子表格会写入所属父单元格的 `nested_tables`。跨页连续表格会根据相同表头和序号列尝试继承上一页的序号。

## 输出说明

### `document.structured.html`

用于人工查看最终表格层级。内嵌图片会显示为“这里有一个图片”的本地超链接；链接会按照图片在单元格中的纵向位置插入文字上下文。

### `document.structured.long.json`

完整调试版本，包含表格、单元格、图片的 `bbox` 等几何信息。适合排查表格归属、嵌套关系和阅读顺序，不建议直接输入生成模型。

### `document.structured.short.json`

生成模型输入版本：

- 删除全部 `bbox`；
- 省略值为空数组的 `nested_tables`；
- 单元格只使用 `content` 表达正文和图片，不重复保存 `text`、`images`；
- 文本项简写为 `{"text": "..."}`；
- 图片项简写为 `{"type": "image", "src": "images/..."}`。

表格层级和文本、图片顺序仍然保留，但 token 数明显少于长版。

### `document.cells.html`

这是 PDF 到结构化文档之间的调试中间产物，不是生成模型输入。它按原坐标绘制检测到的矢量直线、闭合单元格、文字 bbox 和页面兜底区域，主要用于判断：

- 表格线是否检测完整；
- 某段文字被分配到了哪个方框；
- 图片位置是否落入正确的单元格；
- 表格结构错误来自 PDF 几何提取阶段，还是后续结构化阶段。

正常使用时只需读取三个 `document.structured.*` 文件；遇到版面识别问题时再打开 `document.cells.html`。

## 图片文件

PDF 内嵌图片保存在输出目录的 `images/` 中。JSON 和 HTML 使用相对路径引用这些图片，因此移动结果时应一起复制整个输出目录，不要只复制单个 JSON 或 HTML。
