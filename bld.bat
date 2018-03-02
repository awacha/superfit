rem set MENU_DIR="%PREFIX%"\Menu
rem mkdir "%MENU_DIR%"
rem copy menu_temcirclefind.json %MENU_DIR%\
rem if errorlevel 1 exit 1
rem copy tem_circlefind.ico %MENU_DIR%\
rem if errorlevel 1 exit 1
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
