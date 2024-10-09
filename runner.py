from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import argparse
import boto3
import glob
import os
import random
import torch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--model',
    default='./stable-diffusion-xl-base-1.0',
    help='生成に使用するモデルを指定します。',
)
arg_parser.add_argument(
    '--variant',
    default='fp16',
    help='モデルファイルのバリアントを指定します。 none, fp16, ...',
)
arg_parser.add_argument('--batch', type=int, default=1, help='生成回数を指定します。')
arg_parser.add_argument('--num', type=int, default=1, help='生成枚数を指定します。')
arg_parser.add_argument(
    '--output',
    default='/opt/artifact',
    help='出力先ディレクトリを指定します。',
)
arg_parser.add_argument(
    '--prefix',
    default='sdxl-',
    help='出力ファイルのプレフィックスを指定します。',
)
arg_parser.add_argument(
    '--prompt',
    default='An astronaut riding a green horse',
    help='プロンプトを指定します。',
)
arg_parser.add_argument(
    '--negative',
    default='',
    help='ネガティブプロンプトを指定します。',
)
arg_parser.add_argument(
    '--seed',
    type=int,
    default=-1,
    help='乱数シードを指定します。ある値で固定すると同様の構図のイラストが出力されやすくなります。',
)
arg_parser.add_argument(
    '--steps',
    type=int,
    default=20,
    help='サンプリングステップ数を指定します。',
)
arg_parser.add_argument(
    '--width',
    type=int,
    default=1024,
    help='出力画像の幅を指定します。',
)
arg_parser.add_argument(
    '--height',
    type=int,
    default=1024,
    help='出力画像の高さを指定します。',
)
arg_parser.add_argument('--s3-bucket', help='S3のバケットを指定します。')
arg_parser.add_argument('--s3-endpoint', help='S3互換エンドポイントのURLを指定します。')
arg_parser.add_argument('--s3-secret', help='S3のシークレットアクセスキーを指定します。')
arg_parser.add_argument('--s3-token', help='S3のアクセスキーIDを指定します。')

args = arg_parser.parse_args()

seed = int(args.seed) if int(args.seed) >= 0 else random.randint(
    1,
    0x7fffffffffffffff - int(args.num) * int(args.batch),
)

random.seed(seed)

if args.variant != 'single':
    pipe = DiffusionPipeline.from_pretrained(
        args.model if args.model else './stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant=args.variant if args.variant != 'none' else None,
    )
else:
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.model,
        torch_dtype=torch.float16
    )

pipe.to('cuda')

for batch_iteration in range(int(args.batch)):
    print('Seed is {0}'.format(seed))

    images = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        generator=torch.Generator('cuda').manual_seed(seed),
        guidance_scale=7.5,
        height=int(args.height),
        num_images_per_prompt=int(args.num),
        num_inference_steps=int(args.steps),
        output_type='pil',
        width=int(args.width),
    ).images

    for i in range(len(images)):
        images[i].save(
            os.path.join(
                args.output,
                '{}_{}.png'.format(args.prefix, seed + i),
            ),
        )

    seed = seed + len(images)

if args.s3_token and args.s3_secret and args.s3_bucket:
    print('Start uploading to S3')

    s3 = boto3.client(
        's3',
        endpoint_url=args.s3_endpoint if args.s3_endpoint else None,
        aws_access_key_id=args.s3_token,
        aws_secret_access_key=args.s3_secret,
    )

    files = glob.glob(os.path.join(args.output, '*.png'))
    for file in files:
        print(os.path.basename(file))

        s3.upload_file(
            Filename=file,
            Bucket=args.s3_bucket,
            Key=os.path.basename(file),
        )