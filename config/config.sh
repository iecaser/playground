echo adding new user...
adduser zxf
echo setting password...
passwd zxf
asdfqwer1234
echo auth...
vim /etc/sudoers
echo swithing user...
su zxf
echo making working directory...
mkdir /export/workspace
mkdir /export/data
mkdir /export/downloads
mkdir /export/install
cd ~
echo making soft links...
ln -s /export/workspace ~/workspace
ln -s /export/data ~/data
ln -s /export/downloads ~/downloads
ln -s /export/install ~/install
cd downloads
echo downloading emacs...
wget http://gnu.mirrors.hoobly.com/emacs/emacs-26.1.tar.gz
echo extracting emacs...
tar -zxf emacs-26.1.tar.gz
echo installing emacs...
mv emacs-26.1 ~/install/
cd emacs-26.1
./autogen.sh
./configure --with-xpm=no --with-gif=no --with-tiff=no
make
sudo make install
asdfqwer1234

echo git cloning my spacemacs...
cd ~
git clone https://github.com/iecaser/spacemacs.git
mv spacemacs .emacs.d
ln -s ~/.emacs.d/spacemacs ~/.spacemacs



