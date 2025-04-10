


## 六.整理并提交GitHub
0.导出yolo标签时, 背景图要创建一个空文件(因yolo训练时,背景图也需要有对应的标注文件)
1.修改标签显示时的坐标值, 原位置容易遮挡视线，不容易看清边界，故做了调整
2.修改LabelFile类, 可以不加载图片, 只加载label, 提升批量自动标注的运行速度
3.加载语言文件时, 没有发现路径languages/translations/, 但找到了路径resources/translations/, 故改成了这个路径
4.保存子图的改进
	1)将背景图也收集到一起(子图用于发现标定错误的, 背景图用于发现没有标定的)
	2)按权重划分二级文件夹(标定错误的样本, 权重比较低, 只需逐一检查低权重文件夹中的子图, 即可发现标定错误的, 可极大提升人工检查的速度)
5.关于归并类的处理
	说明: 归并类（例如: 动物 = 猫 + 狗)是其它子类的合并， 人工标定时无需关心， 故导入时需过滤掉, 导出时需补充上
	导入导出: 导入时,过滤掉归并类; 导出时,补充上归并类
	自动处理: 标定时,过滤掉归并类; 查错时,要分析归并类, 以便于判定归并类是否有问题
6.将自动推理下的处理任务分拆为四个独立的函数, 并通过下拉框控制当前执行哪个任务
	1)在auto_labeling.ui中增加了一个下拉框，用于选择执行哪个批量处理任务
	2)在auto_labeling.py中添加了对新增下拉框的显示控制和响应事件
	3)给LabelingWidget类增加crop_and_save()函数 和 save_labels_autolabeling()函数
	4)新增的下拉框, 默认选择0, 即: 原来的自动标注功能(保持和原来一样, 未做改动)
	5)新写了:批量自动标定new_shapes_from_auto_labeling_Extend()函数
	6)新写了:提取疑错标签new_shapes_from_auto_labeling_FindErr()函数
	7)新写了:极小样本擦除new_shapes_from_auto_labeling_Erase()函数, 尚未写完
7.进一步对自动推理流程做加速(CPU版在i7-8700K上可以跑到100ms/张了)
	1)对save_labels_autolabeling函数做改进, 最大限度的提升速度
	2)list太慢了, 1万张图耗时约10ms, 为了加速换成了tuple
	3)为了加速, 未使用batch_load_file函数, 而是只加载标签文件
	4)为了关注耗时是否正常, 增加了进度和耗时打印的代码
	5)对new_shapes_from_auto_labeling_Extend函数做了精简, 去掉没用的代码
	6)对new_shapes_from_auto_labeling_FindErr函数做了精简, 去掉没用的代码

## 五.推理加速的修改
1. 增加了一些耗时打印, 通过这些打印发现: 耗时主要在load_file()和to_rgb_cv_img()两个函数中  
2. 通过测试发现: 推理时, 发现根本不需要调用load_file函数, 只需执行其中的五句就行了
3. 通过测试发现: 推理时, 发现根本不需要调用progress_dialog.setValue()函数
4. 保存label时, 若没有图片数据, 则基于图片路径读取宽和高; 否则, 使用图片数据的宽高
5. 修改LabelFile类, 让其支持只加载标签数据, 不加载图片数据
6. 修改save_labels函数, 若没有图片数据, 则基于图片路径读取宽和高; 否则, 使用图片数据的宽高
7. 自动标注时, 不绘制目标框 和 侧边栏中的复选框, 以加快速度
8. 让推理时也可以切子图, 方便标签的验证和矫正，需要点击【重置跟踪器】按钮
9. 涉及到的文件: label_widget.py, yolo.py, canvas.py

run_all_images									
	show_progress_dialog_and_process
		process_next_image(递归)											138ms
			load_file											88ms	
			model_manager.predict_shapes()						45ms
					predict_shapes        
						to_rgb_cv_img	 				 30ms
						preprocess						   5ms
						inference						   9ms
						postprocess						   0ms
					self.new_auto_labeling_result.emit
					== AutoLabelingWidget的new_shapes_from_auto_labeling方法, 计算IOU并做判定
						new_shapes_from_auto_labeling() 
							crop_and_save()
							calculate_iou()
							判定标签的正确性
							set_dirty()	按要求保存label
								save_labels(label_file)
									label_file.save()
			progress_dialog.setValue()							3.7ms

## 四.与保存子图的修改
1. 保存子图时, 文件名中应该包含原图像的文件名, 方便后续的子图和原图的对应
2. 保存子图时, 不支持中文路径的问题
3. 保存子图时, 按置信度进行分类存放
4. 仅自动标注时, 可以保存子图, 需要增加一个复选框(现在借用的跟踪重置按钮), 默认不保存子图
5. 子图的目录应是固定的位置, 不能随原图的路径变化而变化
6. 保存子图时, 也将背景图收集到一起
7. 保存子图时, 先按错误类别分文件夹, 然后再按权重分文件夹
8. 涉及到的文件: image_dialog.py

## 三.与检验有关的修改
1) 增加检测结果与原有标签的iou计算, 并找出: "0-正确的", "1-多检的", "2-位置偏", "3-仅归并", "4-难区分", "5-无归并", "6-少检的"
2) 转存疑似错误的六种标签和图片
3) 转存时, 将 多检 和 偏移 的标签和坐标也写入json文件中, 方便人工矫正
4) 将 保留现有标签 改成了 提取疑错标签, 还不能用, 可能需要做本地化处理  
5) 意思错误标签的目录应是固定的位置, 不能随原图而变, 即: 标签和图片都应放到根目录下的子目录中
6) 涉及到的文件: label_widget.py, zh_CN.ts

## 二.与导出有关的修改
1) 导出yolo标签时, 对齐到小数点后6位
2) 导出时, 同时补充上归并类的标签: LunAll, ...
3) 涉及到的文件: label_converter.py

## 一.与GPU有关的配置(Win7系统)

1) 查看CUDA和Python版本  
    NVIDIA-SMI 560.94  
	Driver Version: 560.94  
	CUDA Version: 12.6  
	Python 3.12.8  

2) 安装requirements  
   pip install -r requirements-gpu-dev.txt

3) 安装ONNXRuntime-GPU  
   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

4) 修改配置文件  
   E:\DLCode\X-AnyLabeling\x-anylabeling-win-gpu.spec
   E:\DLCode\X-AnyLabeling\anylabeling\app_info.py

5) 安装pytorch  
   用迅雷从官网下载torch-2.5.1+cu124-cp312-cp312-win_amd64.whl  
   https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl#sha256=3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6

6) 涉及到的文件: app_info.py, spec文件