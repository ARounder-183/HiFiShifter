HiFiShifter for Linux — portable bundle
=======================================

This folder holds the same app, plug-ins and model library the distribution packages install,
but nothing here needs root: everything lands in your own home folder.

Install
-------

    ./install-user.sh

It asks once and prints every file it writes. Unattended:

    ./install-user.sh --yes

Defaults: the app in ~/.local/lib/hifishifter, the hifishifter command in ~/.local/bin, the
plug-ins in ~/.vst3 and ~/.clap, and the shared weights in $XDG_DATA_HOME/HiFiShifter/models.
Use --prefix=DIR to put the app files elsewhere, or --components=app,models,vst3,clap,menu,mime
to install only some of them (the default is everything this bundle contains).

Uninstall
---------

    ~/.local/bin/hifishifter-uninstall

or ./uninstall-user.sh from this folder after installing. It removes what the installer wrote and
keeps your settings, projects, audio files and models-user.

Requirements
------------

WebKitGTK 4.1 must be provided by the system (Arch/CachyOS, Ubuntu 24.04+, Fedora 39+,
Debian 13+, openSUSE Tumbleweed). Ubuntu 22.04 only ships webkit2gtk-4.0 and is not supported.

Prefer the native packages?
---------------------------

On Debian/Ubuntu install the .deb files, on Fedora/RHEL the .rpm files: the package manager then
handles dependencies, upgrades and removal for you. This bundle exists for everything else.
