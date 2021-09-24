# CCF BDCI 剧本角色情感识别


	#将baseline程序is_train改成True就是启动训练，此版本训练是多gpu的，如果但gpu训练可以稍微改下
	
	python baseline就能启动训练
	
	#模型预测就是is_train改成False然后把gpu指定改成单卡就可以进行推理了。

	python baseline能启动推理
	
	自动调用model_predict函数生成result.tsv文件，支持修改推理不同的文件。
	
	