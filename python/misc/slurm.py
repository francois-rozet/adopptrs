HEADER = """#!/usr/bin/env bash
#
#SBATCH --job-name={name}
#SBATCH --output={name}.out

#SBATCH --ntasks=1
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

SCRATCH=/scratch/$USER/california/

if [[ -d $SCRATCH ]]; then
    echo "$SCRATCH already exists"
else
    mkdir -p $SCRATCH
    cp -r /home/frozet/resources/california/* $SCRATCH
fi

cd ..

source miniconda3/etc/profile.d/conda.sh
conda activate adopptrs

cd adopptrs/python/

"""

JOB = 'python train.py -m {model} {multi} -n {name}_{fold} -e {epochs} -r {resume} -p $SCRATCH -d ../products/models/ -o ../products/{name}_{fold}.txt -s ../products/csv/{name}.csv -f {fold}'

time = '2-00:00:00'
epochs = 20
resume = 0

for model in ['unet', 'segnet']:
	for multi in [False, True]:
		name = '{}'.format(('multi' if multi else '') + model)

		slurm = HEADER.format(
			name=name,
			time=time
		)

		for fold in range(5):
			slurm += JOB.format(
				name=name,
				model=model,
				multi=('-multitask' if multi else ''),
				epochs=epochs,
				resume=resume,
				fold=fold
			) + '\n'

		with open(name + '.sh', 'w') as f:
			f.write(slurm)
