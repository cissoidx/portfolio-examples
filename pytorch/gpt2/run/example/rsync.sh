#! /bin/bash
# rsync -av  --exclude 'report*' ./ 172.21.16.66:$(pwd)
# rsync -av  --exclude 'report*' ./ 172.21.16.67:$(pwd)
# rsync -av  --exclude 'report*' ./ 172.21.16.59:$(pwd)
# rsync -av --exclude 'docs' /localdata/chaon/sdk/2.3.1+793-poptorch 172.21.16.66:/localdata/chaon/sdk/
# rsync -av /localdata/chaon/scripts/2.3.1+793-poptorch.sh 172.21.16.66:/localdata/chaon/scripts/
# rsync -v /localdata/chaon/tmp/3509742919185692277.popart lr17-1-poplar-5:/localdata/chaon/tmp/

# for i in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
# do
    # echo "################### $i ###################"
    # ssh chaon@lr17-1-poplar-$i "mkdir /localdata/chaon"
    # for j in sdk env tmp cachedir scripts repo
    # do
    #     ssh chaon@lr17-1-poplar-$i "mkdir /localdata/chaon/$j"
    # done

    # rsync -av  --exclude 'logs' /localdata/chaon/repo/GPT2 lr17-1-poplar-$i:/localdata/chaon/repo/
    # rsync -av --exclude 'docs' /localdata/chaon/sdk/2.3.0-EA.1+712-poptorch lr17-1-poplar-$i:/localdata/chaon/sdk/
    # rsync -av /localdata/chaon/scripts/2.3.0-EA.1+712-poptorch.sh lr17-1-poplar-$i:/localdata/chaon/scripts/

    # ssh chaon@lr17-1-poplar-$i "virtualenv -p /usr/bin/python3 /localdata/chaon/env/2.3.0-EA.1+712-poptorch"
    # ssh chaon@lr17-1-poplar-$i "/localdata/chaon/env/2.3.0-EA.1+712-poptorch/bin/pip install /localdata/chaon/sdk/2.3.0-EA.1+712-poptorch/poptorch-2.3.0+24683_e4ce21eb49_ubuntu_18_04-cp36-cp36m-linux_x86_64.whl"
    # ssh chaon@lr17-1-poplar-$i "/localdata/chaon/env/2.3.0-EA.1+712-poptorch/bin/pip install horovod transformers tqdm wandb onnx"
    # ssh chaon@lr17-1-poplar-$i "rm -rf /localdata/chaon/tmp/*.lock"
    # echo "##########################################"
# done


for i in 56 57 58 59 61 62 63 64
do
    echo "################### $i ###################"
    # rsync -av  --exclude 'logs' /localdata/chaon/repo/GPT2 172.21.16.$i:/localdata/chaon/repo/
    # ssh chaon@172.21.16.$i "rm -rf /localdata/chaon/tmp/*"
    echo "##########################################"
done