xmodmap ~/workspace/playground/config/keyboardmap/CapsLockSuper.xmodmap
xcape -e 'Mode_switch=Escape'
sudo sslocal -c /etc/shadowsocks/config.json -d start
nohup ~/cerebro/cerebro-0.3.1-x86_64.AppImage > /dev/null 2>&1 &
nohup guake > /dev/null 2>&1 &
