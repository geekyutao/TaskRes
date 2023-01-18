###################### 1-Shot ######################
bash scripts/taskres/main.sh stanford_cars adam_lr2e-3_B256_ep100 ./strong_base/StanfordCars/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh oxford_flowers adam_lr2e-3_B256_ep100 ./strong_base/OxfordFlowers/rn50_1shots/model.pth.tar 1 1.0
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh dtd adam_lr2e-3_B256_ep100 ./strong_base/DescribableTextures/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh fgvc_aircraft adam_lr2e-3_B256_ep100 ./strong_base/FGVCAircraft/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh oxford_pets adam_lr2e-3_B256_ep100 ./strong_base/OxfordPets/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh food101 adam_lr2e-3_B256_ep100 ./strong_base/Food101/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh ucf101 adam_lr2e-3_B256_ep100 ./strong_base/UCF101/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh eurosat adam_lr2e-3_B256_ep100 ./strong_base/EuroSAT/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh sun397 adam_lr2e-3_B256_ep100 ./strong_base/SUN397/rn50_1shots/model.pth.tar 1 0.5
bash scripts/taskres/main.sh imagenet adam_lr2e-4_B256_ep100 ./strong_base/ImageNet/rn50_1shots/model.pth.tar 1 0.5

###################### 2-Shot ######################
bash scripts/taskres/main.sh stanford_cars adam_lr2e-3_B256_ep100 ./strong_base/StanfordCars/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh oxford_flowers adam_lr2e-3_B256_ep100 ./strong_base/OxfordFlowers/rn50_2shots/model.pth.tar 2 1.0
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh dtd adam_lr2e-3_B256_ep100 ./strong_base/DescribableTextures/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh fgvc_aircraft adam_lr2e-3_B256_ep100 ./strong_base/FGVCAircraft/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh oxford_pets adam_lr2e-3_B256_ep100 ./strong_base/OxfordPets/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh food101 adam_lr2e-3_B256_ep100 ./strong_base/Food101/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh ucf101 adam_lr2e-3_B256_ep100 ./strong_base/UCF101/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh eurosat adam_lr2e-3_B256_ep100 ./strong_base/EuroSAT/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh sun397 adam_lr2e-3_B256_ep100 ./strong_base/SUN397/rn50_2shots/model.pth.tar 2 0.5
bash scripts/taskres/main.sh imagenet adam_lr2e-4_B256_ep100 ./strong_base/ImageNet/rn50_2shots/model.pth.tar 2 0.5

###################### 4-Shot ######################
bash scripts/taskres/main.sh stanford_cars adam_lr2e-3_B256_ep100 ./strong_base/StanfordCars/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh oxford_flowers adam_lr2e-3_B256_ep100 ./strong_base/OxfordFlowers/rn50_4shots/model.pth.tar 4 1.0
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep100 ./strong_base/Caltech101/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh dtd adam_lr2e-3_B256_ep100 ./strong_base/DescribableTextures/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh fgvc_aircraft adam_lr2e-3_B256_ep100 ./strong_base/FGVCAircraft/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh oxford_pets adam_lr2e-3_B256_ep100 ./strong_base/OxfordPets/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh food101 adam_lr2e-3_B256_ep100 ./strong_base/Food101/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh ucf101 adam_lr2e-3_B256_ep100 ./strong_base/UCF101/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh eurosat adam_lr2e-3_B256_ep100 ./strong_base/EuroSAT/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh sun397 adam_lr2e-3_B256_ep100 ./strong_base/SUN397/rn50_4shots/model.pth.tar 4 0.5
bash scripts/taskres/main.sh imagenet adam_lr2e-4_B256_ep100 ./strong_base/ImageNet/rn50_4shots/model.pth.tar 4 0.5

###################### 8-Shot ######################
bash scripts/taskres/main.sh stanford_cars adam_lr2e-3_B256_ep200 ./strong_base/StanfordCars/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh oxford_flowers adam_lr2e-3_B256_ep200 ./strong_base/OxfordFlowers/rn50_8shots/model.pth.tar 8 1.0
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep200 ./strong_base/Caltech101/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh dtd adam_lr2e-3_B256_ep200 ./strong_base/DescribableTextures/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh fgvc_aircraft adam_lr2e-3_B256_ep200 ./strong_base/FGVCAircraft/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh oxford_pets adam_lr2e-3_B256_ep200 ./strong_base/OxfordPets/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh food101 adam_lr2e-3_B256_ep200 ./strong_base/Food101/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh ucf101 adam_lr2e-3_B256_ep200 ./strong_base/UCF101/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh eurosat adam_lr2e-3_B256_ep200 ./strong_base/EuroSAT/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh sun397 adam_lr2e-3_B256_ep200 ./strong_base/SUN397/rn50_8shots/model.pth.tar 8 0.5
bash scripts/taskres/main.sh imagenet adam_lr2e-4_B256_ep200 ./strong_base/ImageNet/rn50_8shots/model.pth.tar 8 0.5

###################### 16-Shot ######################
bash scripts/taskres/main.sh stanford_cars adam_lr2e-3_B256_ep200 ./strong_base/StanfordCars/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh oxford_flowers adam_lr2e-3_B256_ep200 ./strong_base/OxfordFlowers/rn50_16shots/model.pth.tar 16 1.0
bash scripts/taskres/main.sh caltech101 adam_lr2e-3_B256_ep200 ./strong_base/Caltech101/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh dtd adam_lr2e-3_B256_ep200 ./strong_base/DescribableTextures/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh fgvc_aircraft adam_lr2e-3_B256_ep200 ./strong_base/FGVCAircraft/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh oxford_pets adam_lr2e-3_B256_ep200 ./strong_base/OxfordPets/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh food101 adam_lr2e-3_B256_ep200 ./strong_base/Food101/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh ucf101 adam_lr2e-3_B256_ep200 ./strong_base/UCF101/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh eurosat adam_lr2e-3_B256_ep200 ./strong_base/EuroSAT/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh sun397 adam_lr2e-3_B256_ep200 ./strong_base/SUN397/rn50_16shots/model.pth.tar 16 0.5
bash scripts/taskres/main.sh imagenet adam_lr2e-4_B256_ep200 ./strong_base/ImageNet/rn50_16shots/model.pth.tar 16 0.5