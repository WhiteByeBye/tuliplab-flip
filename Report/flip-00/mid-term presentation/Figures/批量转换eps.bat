@echo off
setlocal enabledelayedexpansion
echo �뽫��Ҫ�����jpg�ļ����ڱ��ű�Ŀ¼��
echo ���������λ�ڵ�ǰ�ļ���
echo;
pause
echo ��ʼת��...
echo;

if not exist log\ md log\
if exist log\list.txt del log\list.txt

dir /a/b *.png > log\list.txt

set /a count = 0
for /f "delims=." %%i in (log\list.txt) do (
    if exist %%i.png (
        bmeps -c %%i.png %%i.eps
        set /a count += 1
    )
)

echo ת�����, ������%count%���ļ�
echo; 
rd /s/q log
pause