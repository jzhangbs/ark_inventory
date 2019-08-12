# Arknights Inventory Scanner

基于图像识别的明日方舟仓库物品扫描。程序会直接从adb直接获取游戏画面。检测支持除信物、龙门币、原石碎片、至纯源石、合成玉、家具零件外的物品种类和数量。精英化所需材料部分可输出至供[明日方舟工具箱（by一只灰猫）](https://ak.graueneko.xyz/akevolve.html)使用的uri。

## 环境需求

- adb，并需要在PATH中。如果不想改PATH，请修改`get_scene.py:12`的命令。
- tesseract,
    - 安装5.0版本：https://github.com/UB-Mannheim/tesseract/wiki
    - 从https://github.com/tesseract-ocr/tessdata下载`eng.traineddata(22M)`并放在`<安装目录>/tessdata`。
    - 修改`number_recog.py:8`处的路径

## 使用方法

- 手动连接adb。例如MuMu模拟器:
```
$ adb connect localhost:7555
```
- 执行`main.py`。程序会不停地获取画面并执行，每次会将当前屏幕中的物品和数量计入累积记录，输出记录数大于2的物品，并取每种物品记录的众数作为该物品最终的数量。
- 为了提高准确率，建议在运行过程中持续缓慢拖动画面。

## 其他
- 算法可以看prototype/ark_inference，里面有每一步的结果。目前数字识别还会有零星的错误，欢迎提出更好的方法。