# Fish Detection and Processing API

这是一个使用 Flask 构建的 API，用于处理上传的鱼体图像，检测鱼体并估算其质量及投食量。项目使用 YOLO 模型进行鱼体检测，并应用图像处理技术进行边缘检测和轮廓分析。

## 功能

- 接受上传的鱼体图像
- 使用 YOLO 模型检测鱼体
- 对检测到的鱼体进行边缘检测
- 提取几何特征：面积、周长、长宽比
- 估算鱼体质量及投食量
- 返回处理结果，包括标记图像、边缘检测图像和几何特征

## 安装

1. 克隆项目仓库：

   ```bash
   git clone https://github.com/Sherlock-Holmos/predict.git
   
2. 进入项目目录：

    ```bash
    cd <your-project-directory>
    
3. 创建虚拟环境（可选，推荐使用conda创建）：

    ```bash
    conda create predict python==x.x.x ## 自己修改
    conda activate predict

4. 安装依赖：

    ```bash
    pip install -r requirements.txt

## 使用
1. 启动 Flask 应用：

    ```bash
    python app.py

2. 发送 POST 请求到 /process_fish 端点，上传图像文件。你可以使用工具如 curl 或 Postman 来测试 API。例如，使用 curl：

    ```bash
    curl -X POST http://localhost:5000/process_fish -F "image=@/path/to/your/image.png"

## 注意事项
* 确保 yolov8s.pt 模型文件位于项目目录中，或根据实际路径调整代码。
* 如果使用其他版本的 YOLO 模型或其他图像处理库，可能需要调整 requirements.txt 文件和代码。

## 许可证
MIT 许可证。请参阅 LICENSE 文件以获取更多信息。