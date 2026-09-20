# PDF 入口、核验与批量生成

此功能接入原有 `server.py`，不删改原来的启动参数和推理接口。启动方式仍是原命令，例如：

```bash
python pre/server.py --host 127.0.0.1 --port 8081
```

安装 `pre/requirements.txt` 中的依赖后，打开 `http://127.0.0.1:8081/pdf.html`，或从主页面进入 **PDF 转换与核验**。只做 PDF 提取时不需要启动模型；点击批量生成时才请求原有 vLLM 服务。

使用步骤：

1. 拖入/选择一个或多个 PDF。文件选择框可输入本地路径，不允许网页任意读取服务器路径。
2. 填文件类型（如 `AO工序表`、`工艺指示单`、`排故通知单`）；留空则按文件名匹配。高级设置可覆盖表头和卡片拆分表头。
3. 点击 PDF 转换，在文档标签间切换；左侧是结构化块，右侧是原 PDF 的本地单页渲染，支持上一页/下一页、定位和原 PDF 新窗口打开。
4. 点击某块的“修改原始 JSON”可编辑该块。保存会校验 JSON、`headers/rows`、列编号、嵌套和图片路径；非法内容不会保存，revision 冲突会拒绝覆盖。点击“核验正确”后才生成卡片。
5. 进入用例生成页，卡片带位于原输入框上方。“填入”会完整替换输入框；“查看表格”恢复保存结果；“全部导入并生成”逐张调用原有 Base/LoRA 接口，已完成卡片跳过，失败卡片可重试；“导出所有表格”得到 `ao_results.zip`，解压后每张卡一个 JSON。

表头配置位于 `pdf_extract/file_templates.txt`，格式是 `文件名正则 ::=>:: 表头`；详细规则见 `pdf_extract/README.md`。默认拆卡表头为 `工种,序号,工序内容|序号,项目内容`，它决定哪些表按序号拆成 AO 输入，并不等于提取表头。没有匹配表时，整份正文和图片作为一个文档内容块，不丢弃全文。

环境变量：`AO_PDF_TEMPLATES` 指定表头配置；`AO_PDF_WORK_DIR` 指定上传和结果目录（未设置时使用系统临时目录下的 `ao_pdf_workflow`，避免只读仓库触发 Windows WinError 5）。单个 PDF 上限 40 MiB，转换串行执行。保留 source PDF、图片、中间 HTML、long/short JSON 和原始 short 副本；long JSON 不会伪造人工修改后的旧坐标。浏览器用 IndexedDB 保存卡片结果，请及时导出；服务默认只监听本机，不要直接开放公网。

面向程序的接口包括：`POST /api/pdf/convert`、`GET /api/pdf/{id}`、`PUT /api/pdf/{id}/tables/{index}`、`POST /api/pdf/{id}/approve`、`GET /api/pdf/{id}/files/{name}`、`GET /api/pdf/{id}/preview/{page}.png`、`POST /api/pdf/export-results`。
