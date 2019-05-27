nohup xmodmap ~/workspace/playground/config/keyboardmap/CapsLockSuper.xmodmap > /dev/null 2>&1 &
xcape -e 'Mode_switch=Escape'
nohup ~/cerebro/cerebro-0.3.1-x86_64.AppImage > /dev/null 2>&1 &
nohup guake > /dev/null 2>&1 &
sudo sslocal -c /etc/shadowsocks/config.json -d start
