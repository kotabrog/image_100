from setuptools import setup, find_packages

setup(
	name="image_100",
	version='1.0',
	description='画像処理１００本ノックの結果をライブラリ化しました',
	author='Kota Suzuki',
	author_email='suzuki.kota0331@gmail.com',
	url='https://github.com/kotabrog/image_100.git',
	packages=find_packages(),
	install_requires=open('requirements.txt').read().splitlines(),
)
