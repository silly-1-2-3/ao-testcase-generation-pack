# PDF 网页入口

这是 AO 平台的 PDF 上传、核验和批量生成入口。`pre/` 是旧目录，新的部署应使用本目录文件。

从仓库根目录执行：

```bash
pip install -r web/requirements.txt
python web/server.py --host 127.0.0.1 --port 8081
```

打开 `http://127.0.0.1:8081/pdf.html`。服务保留原有 Base/LoRA 推理、检索和 Excel 接口；PDF 页面支持多文件上传、原 PDF 单页预览、short JSON 人工修改、核验后拆卡、批量生成和 ZIP 导出。

PDF 规则和表头配置见 [`pdf_extract/README.md`](../pdf_extract/README.md) 与 `pdf_extract/file_templates.txt`。提取表头和拆卡表头是两套配置：前者由 `file_templates.txt` 控制，后者由独立的 `card_templates.txt` 控制，也可在页面高级设置中覆盖。默认拆卡表头是 `工种,序号,工序内容|序号,项目内容`：外层工步和文本生成一张主卡，嵌套表的每个序号再生成独立子卡。

转换完成后即可下载 cells HTML、structured HTML、long JSON、short JSON 和图片，不必核验、不必生成卡片；页面中的“允许拆分 AO 卡片”复选框默认开启，关闭时只保存中间输出。已导入文档标签上的 `×` 会删除该文档及服务器中间文件，“清空已导入”会删除全部已导入文档。

运行目录默认在系统临时目录 `%TEMP%/ao_pdf_workflow`，避免只读仓库触发 Windows `WinError 5`；可用 `AO_PDF_WORK_DIR` 指定其他可写目录。真实模型未启动时可以查看和核验 PDF，但生成按钮需要本地 vLLM。
