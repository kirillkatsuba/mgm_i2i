"""Balanced online train crops and immutable, image-weighted evaluation crops."""
from collections import Counter
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_mgm_training import script
from mgm_i2i.cluster.data import RandomImageCrops, training_dataset, MULTICROP_PROTOCOL, SampleRange, validate_data_protocol
from mgm_i2i.cluster.crop_sampling import random_crop_coordinates
from mgm_i2i.cluster.metrics import aggregate_images, evaluation_bank
from mgm_i2i.cluster.engine import fingerprint
from mgm_i2i.cluster.budget import TrainingBudget


@pytest.fixture
def full_images(tmp_path):
    entries = []
    for index in range(3):
        image = np.zeros((140, 150, 3), dtype=np.uint8)
        image[..., 0] = np.arange(140)[:, None]
        image[..., 1] = np.arange(150)[None, :]
        image[..., 2] = index
        np.save(tmp_path/f'{index}.npy', image)
        entries.append(dict(image=f'{index}.png', path=f'{index}.npy'))
    manifest = dict(preparation=dict(protocol=MULTICROP_PROTOCOL, bank_size=128, patches_per_image=1,
                                    eval_crops_per_image=5, eval_crop_policy='fixed_random_without_replacement'),
                    train_images=entries, banks={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_each_train_photo_has_four_distinct_unaugmented_crops_per_epoch(full_images):
    dataset = RandomImageCrops(full_images, seed=7, crops_per_image=4)
    assert len(dataset) == 12
    first_epoch = []
    for epoch in range(2):
        seen = []
        for index in range(epoch*12, (epoch+1)*12):
            patch = dataset[index]
            top, left, photo = patch[:, 0, 0].tolist()
            seen.append((photo, top, left))
            # No spatial transforms: coordinate ramps remain in their original orientation.
            assert patch[0, -1, 0] == top + 127 and patch[1, 0, -1] == left + 127
        assert Counter(photo for photo, _, _ in seen) == {0:4, 1:4, 2:4}
        for photo in range(3):
            assert len({(top,left) for p,top,left in seen if p==photo}) == 4
        for start in range(0,12,3):
            assert {photo for photo,_,_ in seen[start:start+3]} == {0,1,2}
        if epoch == 0:
            first_epoch = seen
        else:
            assert seen != first_epoch


def test_crop_sequence_is_identical_across_resume_prefetch_batches_and_methods(full_images):
    cfg = dict(data_protocol=MULTICROP_PROTOCOL, crop=128, seed=8, train_crops_per_image=4)
    a = training_dataset(full_images, dict(cfg, method='mgm'))
    b = training_dataset(full_images, dict(cfg, method='esrgan'))
    for index in [100, 20, 0, 11, 12, 9]:
        a[index]  # Arbitrary prefetch/order changes only the cached permutation.
    expected = torch.stack([b[index] for index in range(5,24)])
    for batch in (1,4):
        loader = DataLoader(a, batch_size=batch, sampler=SampleRange(5,24))
        torch.testing.assert_close(torch.cat(list(loader)), expected, rtol=0, atol=0)
    budget = TrainingBudget(dict(cfg, seconds=1, batch_size=4, epochs=2, samples_per_epoch=12), len(a))
    assert budget.target_samples == 24 and budget.metadata(12)['epochs_completed'] == 1


def test_evaluation_coordinates_are_fixed_unique_and_read_without_recropping(full_images):
    manifest = json.loads((full_images/'manifest.json').read_text())
    images, metadata = [], []
    for photo in (0,1):
        image = np.load(full_images/f'{photo}.npy')
        coords = random_crop_coordinates(140,150,128,5,[123,photo,102])
        assert coords == random_crop_coordinates(140,150,128,5,[123,photo,102])
        assert len(set(coords)) == 5
        for index,(top,left) in enumerate(coords):
            images.append(image[top:top+128,left:left+128])
            metadata.append(dict(image=f'{photo}.png', top=top, left=left, crop_index=index))
    np.save(full_images/'val.npy', np.stack(images))
    manifest['banks']['val.npy'] = dict(crops=metadata)
    (full_images/'manifest.json').write_text(json.dumps(manifest))
    cfg = dict(data_root=str(full_images), data_protocol=MULTICROP_PROTOCOL, crop=128, eval_crop=128)
    data, rows, _, _, size = evaluation_bank(cfg, fingerprint(full_images/'manifest.json'), 'val')
    assert len(data) == 10 and size == 128 and rows == metadata
    for index, image in enumerate(images):
        np.testing.assert_array_equal(data[index].numpy(), image.transpose(2,0,1))
    with pytest.raises(ValueError, match='policy differs'):
        validate_data_protocol(full_images, dict(cfg, eval_crops_per_image=4))
    with pytest.raises(ValueError, match='enough distinct'):
        random_crop_coordinates(128,128,128,5,[1])


def test_metric_aggregation_weights_photos_not_number_of_patches():
    rows = [dict(image='a',score=0.), dict(image='a',score=2.), dict(image='b',score=5.)]
    per_image, means = aggregate_images(rows, ['score'])
    assert per_image == [dict(image='a',score=1.), dict(image='b',score=5.)]
    assert means['score'] == 3.  # Flat patch averaging would incorrectly yield 7/3.


def test_selected_50_50_80_plan_targets_and_rejects_old_protocol_reuse(tmp_path, monkeypatch):
    submit = script('cluster_submit')
    destination = tmp_path/'new'
    monkeypatch.setattr(sys, 'argv', ['submit','--run-root',str(destination), '--variants','mgm_critic_noise',
        '--warmup-epochs','50','--train-epochs','80','--field-warmup-epochs','50',
        '--field-initialization','warmup_features', '--mgm-optimizer','authors','--mgm-beta1','.8',
        '--mgm-microbatch-size','32','--grad-accumulation','1'])
    submit.main()
    plan = json.loads((destination/'plan.json').read_text())
    # Even historical plans carrying a memory setting must not emit a RAM flag.
    plan['scheduler']['memory_gb'] = 32
    assert all(not arg.startswith('--mem') for command in submit.commands(plan, destination/'plan.json')
               for arg in command)
    for path in (submit.ROOT/'slurm').glob('*.sbatch'):
        assert all('--mem' not in line for line in path.read_text().splitlines() if line.startswith('#SBATCH'))
    for phase, epochs, samples, steps in [('warmup',50,160000,10000),('field_warmup',50,160000,5000),('train',80,256000,8000)]:
        for job in plan[phase]:
            cfg = job['config']
            assert cfg['data_protocol'] == MULTICROP_PROTOCOL
            assert cfg['train_crops_per_image'] == 4 and cfg['eval_crops_per_image'] == 5
            assert cfg['epochs'] == epochs and cfg['samples_per_epoch'] == 3200
            assert cfg['epochs'] * cfg['samples_per_epoch'] == samples
            assert samples // (cfg['batch_size'] * cfg.get('grad_accumulation',1)) == steps
    source = tmp_path/'old'
    monkeypatch.setattr(sys,'argv',['submit','--run-root',str(source),'--variants','mgm',
                                    '--data-protocol','random128-v3'])
    submit.main()
    monkeypatch.setattr(sys,'argv',['submit','--run-root',str(tmp_path/'incompatible'), '--variants','mgm',
                                    '--reuse-warmup-from',str(source)])
    with pytest.raises(ValueError, match='Source warmup config differs'):
        submit.main()
