#!/bin/bash
# 执行该脚本的命令：./eval.sh "kp20k" "inspec" "nus" "semeval" "--" "paecrf"


# bash脚本实现os.path.join()函数
function joinpath { 
    local IFS="/"
    echo "$*"
}


# 接收参数
datasets=()
models=()
is_models=false

# 处理参数
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    is_models=true
  elif [[ $is_models == false ]]; then
    datasets+=("$arg")
  else
    models+=("$arg")
  fi
done

# 使用参数
echo "Datasets: ${datasets[@]}"
echo "Models: ${models[@]}"


# 测试
echo "Test: $models test.py"
python ./test.py \
-dataset_directorys "$datasets" \
-model_name "$models" \

# 后处理1：将标签转化为单词，按重要程度排序
echo "Post_process: trans_label2word.py"
python ./trans_label2word.py \
-dataset_directorys "$datasets" \
-model_name "$models" \
-batch_size "16"

# 后处理2：将单词分为sk和ck
echo "Post_process: trans2skck.py"
python ./trans2skck.py \
-dataset_directorys "$datasets" \
-model_name "$models" \

# 评估
echo "Evaluate: evaluate_prediction.py"
pred_files=("predictions.txt" "simplekeyword.txt" "complexkeyword.txt")
trg_files=("allkeyword.txt" "allkeyword.txt" "allkeyword.txt")
out_paths=("all" "sk" "ck")


dir="datasets"
for dataset in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        for (( i=0; i<3; i++ )); do
            pred_file=${pred_files[$i]}
            trg_file=${trg_files[$i]}
            out_path=${out_paths[$i]}

            pred_file_path=$(joinpath "$dir" "$dataset" "$model_name" "$pred_file")
            src_file_path=$(joinpath "$dir" "$dataset" "test_src.txt")
            trg_file_path=$(joinpath "$dir" "$dataset" "$trg_file")
            exp_path=$(joinpath "$dir" "$dataset" "$model_name" "$out_path")
            filtered_pred_path=$(joinpath "$dir" "$dataset" "$model_name" "$out_path")
            
            python ./evaluate_prediction.py \
            -pred_file_path "$pred_file_path" \
            -src_file_path "$src_file_path" \
            -trg_file_path "$trg_file_path" \
            -exp_path "$exp_path" \
            -export_filtered_pred \
            -filtered_pred_path "$filtered_pred_path" \
            -disable_extra_one_word_filter \
            -invalidate_unk \
            -all_ks "5" "M" \
            -present_ks "5" "M" \
            -absent_ks "5" "M"
        done
    done
done
