(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(bibtex-completion-bibliography (quote ("~/Dropbox/papers/org-ref.bib")))
 '(company-minimum-prefix-length 1)
 '(display-time-24hr-format t)
 '(display-time-default-load-average nil)
 '(display-time-format "%H:%M:%S")
 '(display-time-interval 1)
 '(display-time-mail-string "")
 '(display-time-mode 1)
 '(evil-want-Y-yank-to-eol nil)
 '(flycheck-flake8-maximum-line-length 100)
 '(global-display-line-numbers-mode t)
 '(global-flycheck-mode t)
 '(xterm-mouse-mode nil)
 '(helm-split-window-default-side (quote right))
 '(hl-todo-keyword-faces
   (quote
    (("TODO" . "#dc752f")
     ("NEXT" . "#dc752f")
     ("THEM" . "#2aa198")
     ("PENDING" . "#2aa198")
     ("PROG" . "#268bd2")
     ("OKAY" . "#268bd2")
     ("DONT" . "#d70000")
     ("FAIL" . "#d70000")
     ("DONE" . "#86dc2f")
     ("NOTE" . "#875f00")
     ("CANCELLED" . "#875f00")
     ("KLUDGE" . "#875f00")
     ("HACK" . "#875f00")
     ("TEMP" . "#875f00")
     ("FIXME" . "#dc752f")
     ("XXX" . "#dc752f")
     ("XXXX" . "#dc752f"))))
 '(imenu-list-auto-resize nil)
 '(imenu-list-position (quote left))
 '(imenu-list-size 0.25)
 '(menu-bar-mode nil)
 '(multi-term-program "/usr/bin/zsh")
 '(nyan-bar-length 24)
 '(org-modules
   (quote
    (org-bbdb org-bibtex org-docview org-eww org-gnus org-habit org-info org-irc org-mhe org-rmail org-w3m org-toc)))
 '(org-ref-bibliography-notes "~/Dropbox/papers/notes.org")
 '(org-ref-default-bibliography (quote ("~/Dropbox/papers/org-ref.bib")))
 '(org-ref-pdf-directory "~/Dropbox/papers/")
 '(org-todo-keyword-faces
   (quote
    (("TODO" . "#dc752f")
     ("NEXT" . "#dc752f")
     ("THEM" . "#2aa198")
     ("WAITING" . "#2aa198")
     ("PROG" . "#268bd2")
     ("OKAY" . "#268bd2")
     ("DONT" . "#d70000")
     ("FAIL" . "#d70000")
     ("DONE" . "#86dc2f")
     ("NOTE" . "#875f00")
     ("CANCELLED" . "#875f00")
     ("KLUDGE" . "#875f00")
     ("HACK" . "#875f00")
     ("TEMP" . "#875f00")
     ("FIXME" . "#dc752f")
     ("XXX" . "#dc752f")
     ("XXXX" . "#dc752f"))))
 '(org-todo-keywords
   (quote
    ((sequence "TODO(t)" "PROG(p)" "WAITING(w@/!)" "|" "DONE(d!/!)" "CANCELLED(c@/!)" "FAIL(f)"))))
 '(package-selected-packages
   (quote
    (mu4e-maildirs-extension mu4e-alert helm-mu sql-indent ox-hugo dap-mode bui tree-mode ranger emojify emoji-cheat-sheet-plus company-emoji company-quickhelp pandoc-mode ox-pandoc lsp-ui lsp-treemacs lsp-python-ms python helm-rtags google-c-style flycheck-rtags disaster cpp-auto-include company-rtags rtags company-c-headers clang-format apropospriate-theme ample-theme subatomic-theme company-reftex web-mode web-beautify tagedit slim-mode scss-mode sass-mode pug-mode prettier-js impatient-mode simple-httpd helm-css-scss haml-mode emmet-mode counsel-css company-web web-completion-data add-node-modules-path yaml-mode visual-fill-column treemacs ht pfuture spaceline powerline pyvenv spinner pdf-tools key-chord org-category-capture alert log4e gntp markdown-mode epc ctable concurrent deferred htmlize parent-mode window-purpose imenu-list request helm-bibtex parsebib gitignore-mode fringe-helper git-gutter+ git-gutter flyspell-correct pos-tip flycheck flx magit transient git-commit with-editor iedit smartparens paredit anzu shrink-path all-the-icons memoize tablist magit-popup projectile counsel swiper ivy pkg-info epl company biblio biblio-core yasnippet auctex anaconda-mode pythonic f dash s ace-window avy helm-core auto-complete popup org-plus-contrib hydra lv evil goto-chg undo-tree bind-map bind-key async helm doom-themes parrot engine-mode dockerfile-mode docker json-mode docker-tramp json-snatcher json-reformat packed package-lint xterm-color shell-pop multi-term eshell-z eshell-prompt-extras esh-help color-theme-sanityinc-tomorrow yasnippet-snippets ws-butler writeroom-mode winum which-key volatile-highlights vi-tilde-fringe uuidgen use-package unfill treemacs-projectile treemacs-evil toc-org symon symbol-overlay string-inflection spaceline-all-the-icons smeargle restart-emacs rainbow-mode rainbow-identifiers rainbow-delimiters pytest pyenv-mode py-isort py-autopep8 popwin pippel pipenv pip-requirements persp-mode pcre2el password-generator paradox overseer orgit org-ref org-projectile org-present org-pomodoro org-mime org-download org-cliplink org-bullets org-brain open-junk-file nameless mwim move-text mmm-mode material-theme markdown-toc magit-svn magit-gitflow macrostep lorem-ipsum live-py-mode link-hint indent-guide importmagic hungry-delete hl-todo highlight-parentheses highlight-numbers highlight-indentation helm-xref helm-themes helm-swoop helm-pydoc helm-purpose helm-projectile helm-org-rifle helm-mode-manager helm-make helm-gitignore helm-git-grep helm-flx helm-descbinds helm-company helm-c-yasnippet helm-ag google-translate golden-ratio gnuplot gitignore-templates gitconfig-mode gitattributes-mode git-timemachine git-messenger git-link git-gutter-fringe git-gutter-fringe+ gh-md fuzzy font-lock+ flyspell-correct-helm flycheck-pos-tip flycheck-package flx-ido fill-column-indicator fancy-battery eyebrowse expand-region evil-visualstar evil-visual-mark-mode evil-unimpaired evil-tutor evil-textobj-line evil-surround evil-org evil-numbers evil-nerd-commenter evil-matchit evil-magit evil-lisp-state evil-lion evil-indent-plus evil-iedit-state evil-goggles evil-exchange evil-ediff evil-cleverparens evil-args evil-anzu eval-sexp-fu elisp-slime-nav editorconfig dumb-jump dotenv-mode doom-modeline diminish diff-hl devdocs define-word cython-mode csv-mode counsel-projectile company-statistics company-auctex company-anaconda column-enforce-mode color-identifiers-mode clean-aindent-mode centered-cursor-mode browse-at-remote blacken auto-yasnippet auto-highlight-symbol auto-dictionary auto-compile aggressive-indent ace-link ace-jump-helm-line ac-ispell)))
 '(pdf-view-midnight-colors (quote ("#b2b2b2" . "#262626")))
 '(proced-auto-update-flag t)
 '(proced-auto-update-interval 15)
 '(spaceline-all-the-icons-eyebrowse-display-name nil)
 '(spaceline-all-the-icons-hide-long-buffer-path t)
 '(spaceline-all-the-icons-icon-set-git-ahead (quote commit))
 '(spaceline-all-the-icons-icon-set-modified (quote toggle))
 '(spaceline-all-the-icons-icon-set-sun-time (quote sun/moon))
 '(spaceline-all-the-icons-icon-set-vc-icon-git (quote github-logo))
 '(spaceline-all-the-icons-icon-set-window-numbering (quote square))
 '(spaceline-all-the-icons-separator-type (quote cup))
 '(truncate-lines t)
 '(xterm-mouse-mode nil))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )
