# (Optional) activate virtualenv / conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fixmatch-fl

# --------- Paths ---------
TRAIN_DIR="../input/ucf-crime-dataset/Train"
TEST_DIR="../input/ucf-crime-dataset/Test"

SCRIPT="fcrimnetBalance.py"
OUT_JSON="results_fixmatch_fedavg_ucfcrime_withF1.json"

# --------- Experiment params ---------
NUM_CLIENTS=5
ROUNDS=100
LOCAL_EPOCHS=2

IMG_SIZE=224
BATCH_SIZE=64
MU=2                 # unlabeled batch multiplier
LABELED_FRAC=0.2     # 20% labeled per client

LR=3e-4
LAMBDA_U=1.0
TAU=0.90

NUM_WORKERS=4
SEED=12

# --------- Run ---------
echo "======================================"
echo "Running FixMatch-FedAvg on UCF-Crime"
echo "Clients: $NUM_CLIENTS | Rounds: $ROUNDS"
echo "Labeled frac: $LABELED_FRAC | mu: $MU"
echo "Output: $OUT_JSON"
echo "======================================"

CUDA_VISIBLE_DEVICES=4 python fcrimnetBalance.py \
  --train_dir /mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Train \
  --test_dir  /mnt/beegfs/home/mdzarifhossa2025/mdhossa/NSF/ucf-crime-dataset/Test \
  --num_clients $NUM_CLIENTS \
  --rounds $ROUNDS \
  --local_epochs $LOCAL_EPOCHS \
  --img_size $IMG_SIZE \
  --batch_size $BATCH_SIZE \
  --mu $MU \
  --labeled_frac $LABELED_FRAC \
  --lr $LR \
  --lambda_u $LAMBDA_U \
  --tau $TAU \
  --num_workers $NUM_WORKERS \
  --seed $SEED \
  --out_json $OUT_JSON

echo "======================================"
echo "Experiment finished."
echo "Results saved to: $OUT_JSON"
echo "======================================"