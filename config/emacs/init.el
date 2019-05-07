;; Added by Package.el.  This must come before configurations of
;; installed packages.  Don't delete this line.  If you don't want it,
;; just comment it out by adding a semicolon to the start of the line.
;; You may delete these explanatory comments.
(package-initialize)
(add-to-list 'package-archives '("melpa" . "http://melpa.org/packages/") t)
(require 'cl)
(defvar zxf/packages '(
		       company
		       monokai-theme
		       hungry-delete
		       ivy
		       counsel
		       swiper
		       evil
		       ) "Default packages")
(setq package-selected-packages zxf/packages)

(defun zxf/packages-installed-p ()
  (loop for pkg in zxf/packages
	when (not (package-installed-p pkg)) do (return nil)
	finally (return t)))
(unless (zxf/packages-installed-p)
  (message "%s" "Refreshing package database...")
  (package-refresh-contents)
  (dolist (pkg zxf/packages)
    (when (not (package-installed-p pkg))
      (package-install pkg))))

(global-company-mode t)
(load-theme 'monokai t)
(tool-bar-mode -1)
(scroll-bar-mode -1)
(global-linum-mode t)
(add-to-list 'default-frame-alist '(fullscreen . maximized))

;; ivy
(ivy-mode 1)
(setq ivy-use-virtual-buffers t)
(setq enable-recursive-minibuffers t)
;; enable this if you want `swiper' to use it
;; (setq search-default-mode #'char-fold-to-regexp)
(global-set-key "\C-s" 'swiper)
(global-set-key (kbd "C-c C-r") 'ivy-resume)
(global-set-key (kbd "<f6>") 'ivy-resume)
(global-set-key (kbd "M-x") 'counsel-M-x)
(global-set-key (kbd "C-x C-f") 'counsel-find-file)
(global-set-key (kbd "<f1> f") 'counsel-describe-function)
(global-set-key (kbd "<f1> v") 'counsel-describe-variable)
(global-set-key (kbd "<f1> l") 'counsel-find-library)
(global-set-key (kbd "<f2> i") 'counsel-info-lookup-symbol)
(global-set-key (kbd "<f2> u") 'counsel-unicode-char)
(global-set-key (kbd "C-c g") 'counsel-git)
(global-set-key (kbd "C-c j") 'counsel-git-grep)
(global-set-key (kbd "C-c k") 'counsel-ag)
(global-set-key (kbd "C-x l") 'counsel-locate)
(global-set-key (kbd "C-S-o") 'counsel-rhythmbox)
(global-set-key (kbd "\C-x \C-b") 'ivy-switch-buffer)
;;(global-set-key (kbd "\C-x \C-x") ' )

(define-key minibuffer-local-map (kbd "C-r") 'counsel-minibuffer-history)

(defun quick-open-init-file()
    (interactive)
  (find-file "~/.emacs.d/init.el"))
(global-set-key (kbd "<f4>") 'quick-open-init-file)

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(company-tooltip-idle-delay 0.2)
 '(custom-safe-themes
   (quote
    ("bd7b7c5df1174796deefce5debc2d976b264585d51852c962362be83932873d9" default))))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )

(add-hook 'emacs-lisp-mode-hook 'show-paren-mode)
(global-hl-line-mode t)
(require 'evil)
(evil-mode 1)
(define-key evil-insert-state-map "\C-h" 'evil-delete-backward-char)
(define-key evil-normal-state-map "\C-u" 'evil-scroll-up)

;; latex
(require 'auctex-latexmk)
(auctex-latexmk-setup)

(with-eval-after-load 'evil
      (defalias #'forward-evil-word #'forward-evil-symbol))
