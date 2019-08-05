
import boto3
import hug
from hug.json_module import json as json_converter
from moto import mock_s3

from soiling.calculator import generate_workbook

BUCKET_NAME = '8me/soiling/8me-soiling-results'


def __gen_error_message(msg, code):
    return dict(code=code, message=msg)


@mock_s3
@hug.default_input_format("application/json")
@hug.post('/calculate', versions=1)
def soiling(body):
    parsed = hug.input_format.json(body)
    wb = generate_workbook(parsed)

    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=BUCKET_NAME)
    obj = s3.Object(BUCKET_NAME, wb.wb_name())
    obj.put(Body=wb.save_workbook_mem())

    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': BUCKET_NAME,
            'Key': wb.wb_name(),
        }
    )
    return dict(result=url)
