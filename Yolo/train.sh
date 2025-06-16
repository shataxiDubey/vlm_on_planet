name=lucknow
task=obb
suffix=v_one_planet_image

model_dir=/home/shataxi.dubey/shataxi_work/vlm_on_planet/Yolo

base_path=/home/shataxi.dubey/shataxi_work/vlm_on_planet/Yolo
data=/home/shataxi.dubey/shataxi_work/vlm_on_planet/Yolo/data.yml
imgsz=640
epochs=100
device=1
experimentName=$name\_$task\_$suffix\_$model\_$imgsz\_$epochs\_batch1


echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Experiment Name: $experimentName"
echo "Model: $model"
echo "Data: $data"
echo "Image Size: $imgsz"
echo "Epochs: $epochs"
echo "Device: $device"

# removing existing directory(before execution code uncomment this line to remove existing directory)
# rm -rf $base_path/runs/$experimentName

nohup yolo obb train model=yolo11m-obb.yaml batch=1 data=$data imgsz=$imgsz device=$device name=$base_path/runs/$experimentName epochs=$epochs val=False save_conf=True save_txt=True save=True > $base_path/$experimentName.log 2>&1 &
echo "Job fired!"
