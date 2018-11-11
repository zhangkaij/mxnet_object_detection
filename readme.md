python train_faster_rcnn.py --gpus 0,2 --network resnet50_v1b --dataset coco --data-path ~/data/coco --save-prefix models/saved/ --resume models/faster_rcnn_resnet50_init.params

## 准备自己的rec文件
假设数据文件的结构为：
+ data_self
	+ 20181107-35
		+ DATA_00001
		- DATA_00001.json
	+ 20181109-01
	...
1. 首先创建软连接把数据放在代码第一级目录中
```
cd code/mxnet_object_detection
ln -s /home/user_name/data/data_self ./
```
2. 创建LST文件
其中img_path从data_self下一级目录开始

    ```
    def write_line(img_path, im_shape, boxes, ids, idx):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        # normalized bboxes (recommanded)
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line
    ```

3. 使用img2rec.py生成对应的rec文件
```
python im2rec.py ./data/train.lst ./data --pass-through --pack-label
```























