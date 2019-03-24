#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

LCtrl & h::
  send {backspace}
  return
LCtrl & [::
  send {esc}
  return
LCtrl & m::
  send {return}
  return

RAlt & h::
  send {left}
  return
RAlt & j::
  send {down}
  return
RAlt & k::
  send {up}
  return
RAlt & l::
  send {right}
  return
RAlt & ,::
  send {home}
  return
RAlt & .::
  send {end}
  return
RAlt & [::
  send {pgup}
  return
RAlt & ]::
  send {pgdn}
  return
RAlt & backspace::
  send {delete}
  return
RAlt & 1::
  send {F1}
  return
RAlt & 2::
  send {F2}
  return
RAlt & 3::
  send {F3}
  return
RAlt & 4::
  send {F4}
  return
RAlt & 5::
  send {F5}
  return
RAlt & 6::
  send {F6}
  return
RAlt & 7::
  send {F7}
  return
RAlt & 8::
  send {F8}
  return
RAlt & 9::
  send {F9}
  return
RAlt & 0::
  send {F10}
  return
RAlt & -::
  send {F11}
  return
RAlt & =::
  send {F12}
  return