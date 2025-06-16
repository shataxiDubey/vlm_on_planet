# name=wb_small_airshed
name=lucknow
task=obb
suffix=v_32_0
root_path=/home/shataxi.dubey/shataxi_work/vlm_on_planet/Yolo
# state_part_name=wb
state_part_name=lucknow

data=/home/shataxi.dubey/shataxi_work/vlm_on_planet/lucknow_train_test_split/test/images
# data=/home/shataxi.dubey/shataxi_work/vlm_on_planet/test/images
imgsz=320
epochs=100
device=0
experimentName=$name\_$task\_$suffix\_$model\_$imgsz\_$epochs
model=/home/shataxi.dubey/shataxi_work/vlm_on_planet/Yolo/runs/lucknow_obb_v_32_0__320_100_batch32/weights/best.pt
log_dir=$root_path/$experimentName/$state_part_name
log_file=$log_dir/$state_part_name.log


echo "Name: $name"
echo "Task: $task"
echo "Suffix: $suffix"
echo "Experiment Name: $experimentName"
echo "Data: $data"
echo "Image Size: $imgsz"
echo "Epochs: $epochs"
echo "Device: $device"

# removing existing directory
# rm -rf $base_path/runs/$experimentName
mkdir -p $log_dir

nohup yolo obb predict model=$model source=$data conf=0.01 imgsz=$imgsz device=$device name=$root_path/predict/$experimentName/$state_part_name save_txt=True save=False save_conf=True save_crop=False verbose=True > $log_file 2>&1 &
echo "Job fired!"
