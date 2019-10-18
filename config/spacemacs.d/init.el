;; -*- mode: emacs-lisp -*-
;; This file is loaded by Spacemacs at startup.
;; It must be stored in your home directory.

(defun dotspacemacs/layers ()
  "Configuration Layers declaration.
You should not put any user code in this function besides modifying the variable
values."
  (setq-default
   ;; Base distribution to use. This is a layer contained in the directory
   ;; `+distribution'. For now available distributions are `spacemacs-base'
   ;; or `spacemacs'. (default 'spacemacs)
   dotspacemacs-distribution 'spacemacs
   ;; Lazy installation of layers (i.e. layers are installed only when a file
   ;; with a supported type is opened). Possible values are `all', `unused'
   ;; and `nil'. `unused' will lazy install only unused layers (i.e. layers
   ;; not listed in variable `dotspacemacs-configuration-layers'), `all' will
   ;; lazy install any layer that support lazy installation even the layers
   ;; listed in `dotspacemacs-configuration-layers'. `nil' disable the lazy
   ;; installation feature and you have to explicitly list a layer in the
   ;; variable `dotspacemacs-configuration-layers' to install it.
   ;; (default 'unused)
   dotspacemacs-enable-lazy-installation 'unused
   ;; If non-nil then Spacemacs will ask for confirmation before installing
   ;; a layer lazily. (default t)
   dotspacemacs-ask-for-lazy-installation t
   ;; If non-nil layers with lazy install support are lazy installed.
   ;; List of additional paths where to look for configuration layers.
   ;; Paths must have a trailing slash (i.e. `~/.mycontribs/')
   dotspacemacs-configuration-layer-path '()
   ;; List of configuration layers to load.
   dotspacemacs-configuration-layers
   '(html
     better-defaults
     bibtex
     docker
     emacs-lisp
     emoji
     git
     helm
     latex
     lsp
     markdown
     search-engine
     spell-checking
     syntax-checking
     yaml
     (org :variables org-want-todo-bindings t
          org-enable-hugo-support t)
     (auto-completion :variables
                      auto-completion-enable-snippets-in-popup t
                      auto-completion-tab-key-behavior 'cycle
                      auto-completion-enable-sort-by-usage t)
     (version-control :variables
                       version-control-diff-tool 'diff-hl
                       version-control-diff-side 'left
                       version-control-global-margin t)
     (shell :variables shell-default-shell 'ansi-term
            shell-default-term-shell "/usr/bin/zsh")
     (python :variables python-backend 'anaconda)
     (c-c++ :variables
            c-c++-backend 'lsp-cquery
            c-c++-default-mode-for-headers 'c++-mode
            c-c++-enable-clang-format-on-save t
            c-c++-enable-auto-newline t
            c-c++-adopt-subprojects t
            c-c++-lsp-sem-highlight-rainbow t
            )
     (colors :variables
             colors-enable-nyan-cat-progress-bar t)
     )
   ;; List of additional packages that will be installed without being
   ;; wrapped in a layer. If you need some configuration for these
   ;; packages, then consider creating a layer. You can also put the
   ;; configuration in `dotspacemacs/user-config'.
   dotspacemacs-additional-packages '((helm-swoop :location (recipe :fetcher github :repo "ashiklom/helm-swoop"))
                                      parrot
                                      ox-pandoc
                                      py-autopep8)
   ;; A list of packages that cannot be updated.
   dotspacemacs-frozen-packages '(
                                  helm-swoop
                                  )
   ;; A list of packages that will not be installed and loaded.
   dotspacemacs-excluded-packages '(evil-escape
                                    avy
                                    yapfify
                                    neotree)
   ;; Defines the behaviour of Spacemacs when installing packages.
   ;; Possible values are `used-only', `used-but-keep-unused' and `all'.
   ;; `used-only' installs only explicitly used packages and uninstall any
   ;; unused packages as well as their unused dependencies.
   ;; `used-but-keep-unused' installs only the used packages but won't uninstall
   ;; them if they become unused. `all' installs *all* packages supported by
   ;; Spacemacs and never uninstall them. (default is `used-only')
   dotspacemacs-install-packages 'used-only))

(defun dotspacemacs/init ()
  "Initialization function.
This function is called at the very startup of Spacemacs initialization
before layers configuration.
You should not put any user code in there besides modifying the variable
values."
  ;; This setq-default sexp is an exhaustive list of all the supported
  ;; spacemacs settings.
  (setq-default
   ;; If non nil ELPA repositories are contacted via HTTPS whenever it's
   ;; possible. Set it to nil if you have no way to use HTTPS in your
   ;; environment, otherwise it is strongly recommended to let it set to t.
   ;; This variable has no effect if Emacs is launched with the parameter
   ;; `--insecure' which forces the value of this variable to nil.
   ;; (default t)
   dotspacemacs-elpa-https t
   ;; Maximum allowed time in seconds to contact an ELPA repository.
   dotspacemacs-elpa-timeout 5
   ;; If non nil then spacemacs will check for updates at startup
   ;; when the current branch is not `develop'. Note that checking for
   ;; new versions works via git commands, thus it calls GitHub services
   ;; whenever you start Emacs. (default nil)
   dotspacemacs-check-for-update t
   ;; If non-nil, a form that evaluates to a package directory. For example, to
   ;; use different package directories for different Emacs versions, set this
   ;; to `emacs-version'.
   dotspacemacs-elpa-subdirectory nil
   ;; One of `vim', `emacs' or `hybrid'.
   ;; `hybrid' is like `vim' except that `insert state' is replaced by the
   ;; `hybrid state' with `emacs' key bindings. The value can also be a list
   ;; with `:variables' keyword (similar to layers). Check the editing styles
   ;; section of the documentation for details on available variables.
   ;; (default 'vim)
   dotspacemacs-editing-style 'vim
   ;; If non nil output loading progress in `*Messages*' buffer. (default nil)
   dotspacemacs-verbose-loading nil
   ;; Specify the startup banner. Default value is `official', it displays
   ;; the official spacemacs logo. An integer value is the index of text
   ;; banner, `random' chooses a random text banner in `core/banners'
   ;; directory. A string value must be a path to an image format supported
   ;; by your Emacs build.
   ;; If the value is nil then no banner is displayed. (default 'official)
   dotspacemacs-startup-banner "~/.spacemacs.d/imgs/ya.gif"
   ;; List of items to show in startup buffer or an association list of
   ;; the form `(list-type . list-size)`. If nil then it is disabled.
   ;; Possible values for list-type are:
   ;; `recents' `bookmarks' `projects' `agenda' `todos'."
   ;; List sizes may be nil, in which case
   ;; `spacemacs-buffer-startup-lists-length' takes effect.
   dotspacemacs-startup-lists '((todos . 5)
                                (bookmarks . 5)
                                (projects . 5)
                                (recents . 5))
   ;; True if the home buffer should respond to resize events.
   dotspacemacs-startup-buffer-responsive t
   ;; Default major mode of the scratch buffer (default `text-mode')
   dotspacemacs-scratch-mode 'org-mode
   ;; List of themes, the first of the list is loaded when spacemacs starts.
   ;; Press <SPC> T n to cycle to the next theme in the list (works great
   ;; with 2 themes variants, one dark and one light)
   dotspacemacs-themes '(doom-one
                         ample
                         doom-vibrant
                         doom-tomorrow-night
                         spacemacs-dark
                         )
   ;; Chose one from followings
   ;; 'spacemacs 'all-the-icons 'vim-powerline 'vanilla
   dotspacemacs-mode-line-theme 'all-the-icons
   ;; dotspacemacs-mode-line-theme 'vanilla
   ;; dotspacemacs-mode-line-theme 'doom
   ;; If non nil the cursor color matches the state color in GUI Emacs.
   dotspacemacs-colorize-cursor-according-to-state t
   ;; Default font, or prioritized list of fonts. `powerline-scale' allows to
   ;; quickly tweak the mode-line size to make separators look not too crappy.
   dotspacemacs-default-font '("DejaVu Sans Mono"
                               :size 20
                               :weight normal
                               :width normal
                               :powerline-scale 1)
   ;; The leader key
   dotspacemacs-leader-key "SPC"
   ;; The key used for Emacs commands (M-x) (after pressing on the leader key).
   ;; (default "SPC")
   dotspacemacs-emacs-command-key "SPC"
   ;; The key used for Vim Ex commands (default ":")
   dotspacemacs-ex-command-key ":"
   ;; The leader key accessible in `emacs state' and `insert state'
   ;; (default "M-m")
   dotspacemacs-emacs-leader-key "C-M-m"
   ;; Major mode leader key is a shortcut key which is the equivalent of
   ;; pressing `<leader> m`. Set it to `nil` to disable it. (default ",")
   dotspacemacs-major-mode-leader-key "C-M-m"
   ;; Major mode leader key accessible in `emacs state' and `insert state'.
   ;; (default "C-M-m")
   dotspacemacs-major-mode-emacs-leader-key "C-M-m"
   ;; These variables control whether separate commands are bound in the GUI to
   ;; the key pairs C-i, TAB and C-m, RET.
   ;; Setting it to a non-nil value, allows for separate commands under <C-i>
   ;; and TAB or <C-m> and RET.
   ;; In the terminal, these pairs are generally indistinguishable, so this only
   ;; works in the GUI. (default nil)
   dotspacemacs-distinguish-gui-tab t
   ;; If non nil `Y' is remapped to `y$' in Evil states. (default nil)
   dotspacemacs-remap-Y-to-y$ nil
   ;; If non-nil, the shift mappings `<' and `>' retain visual state if used
   ;; there. (default t)
   dotspacemacs-retain-visual-state-on-shift t
   ;; If non-nil, J and K move lines up and down when in visual mode.
   ;; (default nil)
   dotspacemacs-visual-line-move-text t
   ;; If non nil, inverse the meaning of `g' in `:substitute' Evil ex-command.
   ;; (default nil)
   dotspacemacs-ex-substitute-global nil
   ;; Name of the default layout (default "Default")
   dotspacemacs-default-layout-name "Default"
   ;; If non nil the default layout name is displayed in the mode-line.
   ;; (default nil)
   dotspacemacs-display-default-layout t
   ;; If non nil then the last auto saved layouts are resume automatically upon
   ;; start. (default nil)
   dotspacemacs-auto-resume-layouts nil
   ;; Size (in MB) above which spacemacs will prompt to open the large file
   ;; literally to avoid performance issues. Opening a file literally means that
   ;; no major mode or minor modes are active. (default is 1)
   dotspacemacs-large-file-size 1
   ;; Location where to auto-save files. Possible values are `original' to
   ;; auto-save the file in-place, `cache' to auto-save the file to another
   ;; file stored in the cache directory and `nil' to disable auto-saving.
   ;; (default 'cache)
   dotspacemacs-auto-save-file-location 'cache
   ;; Maximum number of rollback slots to keep in the cache. (default 5)
   dotspacemacs-max-rollback-slots 5
   ;; If non nil, `helm' will try to minimize the space it uses. (default nil)
   dotspacemacs-helm-resize nil
   ;; if non nil, the helm header is hidden when there is only one source.
   ;; (default nil)
   dotspacemacs-helm-no-header nil
   ;; define the position to display `helm', options are `bottom', `top',
   ;; `left', or `right'. (default 'bottom)
   dotspacemacs-helm-position 'right
   ;; Controls fuzzy matching in helm. If set to `always', force fuzzy matching
   ;; in all non-asynchronous sources. If set to `source', preserve individual
   ;; source settings. Else, disable fuzzy matching in all sources.
   ;; (default 'always)
   dotspacemacs-helm-use-fuzzy 'always
   ;; If non nil the paste micro-state is enabled. When enabled pressing `p`
   ;; several times cycle between the kill ring content. (default nil)
   dotspacemacs-enable-paste-transient-state nil
   ;; Which-key delay in seconds. The which-key buffer is the popup listing
   ;; the commands bound to the current keystroke sequence. (default 0.4)
   dotspacemacs-which-key-delay 0.3
   ;; Which-key frame position. Possible values are `right', `bottom' and
   ;; `right-then-bottom'. right-then-bottom tries to display the frame to the
   ;; right; if there is insufficient space it displays it at the bottom.
   ;; (default 'bottom)
   dotspacemacs-which-key-position 'right-then-bottom
   ;; If non nil a progress bar is displayed when spacemacs is loading. This
   ;; may increase the boot time on some systems and emacs builds, set it to
   ;; nil to boost the loading time. (default t)
   dotspacemacs-loading-progress-bar t
   ;; If non nil the frame is fullscreen when Emacs starts up. (default nil)
   ;; (Emacs 24.4+ only)
   dotspacemacs-fullscreen-at-startup nil
   ;; If non nil `spacemacs/toggle-fullscreen' will not use native fullscreen.
   ;; Use to disable fullscreen animations in OSX. (default nil)
   dotspacemacs-fullscreen-use-non-native nil
   ;; If non nil the frame is maximized when Emacs starts up.
   ;; Takes effect only if `dotspacemacs-fullscreen-at-startup' is nil.
   ;; (default nil) (Emacs 24.4+ only)
   dotspacemacs-maximized-at-startup t
   ;; A value from the range (0..100), in increasing opacity, which describes
   ;; the transparency level of a frame when it's active or selected.
   ;; Transparency can be toggled through `toggle-transparency'. (default 90)
   dotspacemacs-active-transparency 90
   ;; A value from the range (0..100), in increasing opacity, which describes
   ;; the transparency level of a frame when it's inactive or deselected.
   ;; Transparency can be toggled through `toggle-transparency'. (default 90)
   dotspacemacs-inactive-transparency 90
   ;; If non nil show the titles of transient states. (default t)
   dotspacemacs-show-transient-state-title t
   ;; If non nil show the color guide hint for transient state keys. (default t)
   dotspacemacs-show-transient-state-color-guide t
   ;; If non nil unicode symbols are displayed in the mode line. (default t)
   dotspacemacs-mode-line-unicode-symbols t
   ;; If non nil smooth scrolling (native-scrolling) is enabled. Smooth
   ;; scrolling overrides the default behavior of Emacs which recenters point
   ;; when it reaches the top or bottom of the screen. (default t)
   dotspacemacs-smooth-scrolling t
   ;; Control line numbers activation.
   ;; If set to `t' or `relative' line numbers are turned on in all `prog-mode' and
   ;; `text-mode' derivatives. If set to `relative', line numbers are relative.
   ;; This variable can also be set to a property list for finer control:
   ;; '(:relative nil
   ;;   :disabled-for-modes dired-mode
   ;;                       doc-view-mode
   ;;                       markdown-mode
   ;;                       org-mode
   ;;                       pdf-view-mode
   ;;                       text-mode
   ;;   :size-limit-kb 1000)
   ;; (default nil)
   dotspacemacs-line-numbers 'relative
   ;; Code folding method. Possible values are `evil' and `origami'.
   ;; (default 'evil)
   dotspacemacs-folding-method 'evil
   ;; If non-nil smartparens-strict-mode will be enabled in programming modes.
   ;; (default nil)
   dotspacemacs-smartparens-strict-mode nil
   ;; If non-nil pressing the closing parenthesis `)' key in insert mode passes
   ;; over any automatically added closing parenthesis, bracket, quote, etc…
   ;; This can be temporary disabled by pressing `C-q' before `)'. (default nil)
   dotspacemacs-smart-closing-parenthesis nil
   ;; Select a scope to highlight delimiters. Possible values are `any',
   ;; `current', `all' or `nil'. Default is `all' (highlight any scope and
   ;; emphasis the current one). (default 'all)
   dotspacemacs-highlight-delimiters 'all
   ;; If non nil, advise quit functions to keep server open when quitting.
   ;; (default nil)
   dotspacemacs-persistent-server nil
   ;; List of search tool executable names. Spacemacs uses the first installed
   ;; tool of the list. Supported tools are `ag', `pt', `ack' and `grep'.
   ;; (default '("ag" "pt" "ack" "grep"))
   dotspacemacs-search-tools '("ag" "pt" "ack" "grep")
   ;; The default package repository used if no explicit repository has been
   ;; specified with an installed package.
   ;; Not used for now. (default nil)
   dotspacemacs-default-package-repository nil
   ;; Delete whitespace while saving buffer. Possible values are `all'
   ;; to aggressively delete empty line and long sequences of whitespace,
   ;; `trailing' to delete only the whitespace at end of lines, `changed'to
   ;; delete only whitespace for changed lines or `nil' to disable cleanup.
   ;; (default nil)
   dotspacemacs-whitespace-cleanup 'trailing
   ;; Show file path in title bar
   dotspacemacs-frame-title-format "%f"
   ))

(defun dotspacemacs/user-init ()
  "Initialization function for user code.
It is called immediately after `dotspacemacs/init', before layer configuration
executes.
 This function is mostly useful for variables that need to be set
before packages are loaded. If you are unsure, you should try in setting them in
`dotspacemacs/user-config' first."
  ;; (use-package all-the-icons)
  ;; (setq configuration-layer--elpa-archives
  ;;     '(("melpa-cn" . "http://elpa.emacs-china.org/melpa/")
  ;;       ("org-cn"   . "http://elpa.emacs-china.org/org/")
  ;;       ("gnu-cn"   . "http://elpa.emacs-china.org/gnu/")))
  ;; (kill-buffer "*scratch*")
  (setenv "no_proxy" "127.0.0.1,localhost")
  (setenv "NO_PROXY" "127.0.0.1,localhost")
  )

(defun dotspacemacs/user-config ()
  "Configuration function for user code.
This function is called at the very end of Spacemacs initialization after
layers configuration.
This is the place where most of your configurations should be done. Unless it is
explicitly specified that a variable should be set before a package is loaded,
you should place your code here."
  ;; win10 ka
  (when (eq system-type 'windows-nt)
    (setq gc-cons-threshold (* 512 1024 1024))
    (setq gc-cons-percentage 0.5)
    (run-with-idle-timer 5 t #'garbage-collect))
  ;; git token
  (setq org-latex-pdf-process
        '(
          "xelatex -interaction nonstopmode -output-directory %o %f"
          "xelatex -interaction nonstopmode -output-directory %o %f"
          "xelatex -interaction nonstopmode -output-directory %o %f"
          "rm -fr %b.out %b.log %b.tex auto"))
  ;; all icons
  (spaceline-all-the-icons--setup-anzu)            ;; Enable anzu searching
  (spaceline-all-the-icons--setup-package-updates) ;; Enable package update indicator
  (spaceline-all-the-icons--setup-git-ahead)       ;; Enable # of commits ahead of upstream in git
  (spaceline-all-the-icons--setup-paradox)         ;; Enable Paradox mode line
  (parrot-mode)
  (global-centered-cursor-mode nil)
  ;; org
  ;; define the refile targets
  (defvar org-agenda-dir "" "gtd org files location")
  (setq-default org-agenda-dir "~/Dropbox/org-notes")
  (setq org-agenda-file-note (expand-file-name "Notes.org" org-agenda-dir))
  (setq org-agenda-file-bill (expand-file-name "Bill.org" org-agenda-dir))
  (setq org-agenda-file-gtd (expand-file-name "GTD.org" org-agenda-dir))
  (setq org-agenda-file-journal (expand-file-name "Journal.org" org-agenda-dir))
  (setq org-agenda-file-code-snippet (expand-file-name "snippet.org" org-agenda-dir))
  (setq org-default-notes-file (expand-file-name "GTD.org" org-agenda-dir))
  (setq org-agenda-files (list org-agenda-dir))
  (global-set-key (kbd "C-c n")
                  (lambda () (interactive) (find-file org-agenda-file-note)))
  (with-eval-after-load 'org-agenda
    (define-key org-agenda-mode-map (kbd "P") 'org-pomodoro)
    (spacemacs/set-leader-keys-for-major-mode 'org-agenda-mode
      "." 'spacemacs/org-agenda-transient-state/body)
    )
  (defun retrieve-chrome-current-tab-url()
    "Get the URL of the active tab of the first window"
    (interactive)
    (let ((result (do-applescript
                   (concat
                    "set frontmostApplication to path to frontmost application\n"
                    "tell application \"Google Chrome\"\n"
                    "	set theUrl to get URL of active tab of first window\n"
                    "	set theResult to (get theUrl) \n"
                    "end tell\n"
                    "activate application (frontmostApplication as text)\n"
                    "set links to {}\n"
                    "copy theResult to the end of links\n"
                    "return links as string\n"))))
      (format "%s" (s-chop-suffix "\"" (s-chop-prefix "\"" result)))))
  (defun get-year-and-month ()
    (list (format-time-string "%Y - Year") (format-time-string "%m - Month")))
  (defun find-month-tree ()
    (let* ((path (get-year-and-month))
          (level 1)
          end)
      (unless (derived-mode-p 'org-mode)
        (error "Target buffer \"%s\" should be in Org mode" (current-buffer)))
      (goto-char (point-min))             ;移动到 buffer 的开始位置
      ;; 先定位表示年份的 headline，再定位表示月份的 headline
      (dolist (heading path)
        (let ((re (format org-complex-heading-regexp-format
                          (regexp-quote heading)))
              (cnt 0))
          (if (re-search-forward re end t)
              (goto-char (point-at-bol))  ;如果找到了 headline 就移动到对应的位置
            (progn                        ;否则就新建一个 headline
              (or (bolp) (insert "\n"))
              (if (/= (point) (point-min)) (org-end-of-subtree t t))
              (insert (make-string level ?*) " " heading "\n |Time|Type|Description|Money|\n |-+-+-+-|\n"))))
        (setq level (1+ level))
        (setq end (save-excursion (org-end-of-subtree t t))))
      (org-end-of-subtree)))
  (setq org-capture-templates
        '(("t" "Life-Todo" entry (file+headline org-agenda-file-gtd "Life")
           "* TODO [#B] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:END:\n %?"
          :empty-lines 1)
          ("b" "Billing" plain
            (file+function org-agenda-file-bill find-month-tree)
            " | %U | %^{Type} | %^{Description} | %^{Money} |" :kill-buffer t)
          ("e" "English-Todo" entry (file+headline org-agenda-file-gtd "English")
           "* TODO [#B] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:END:\n %?"
           :empty-lines 1)
          ("h" "Hack-Todo" entry (file+headline org-agenda-file-gtd "Hack")
           "* TODO [#C] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:END:\n %?"
           :empty-lines 1)
          ("n" "Notes" entry (file+headline org-agenda-file-note "Quick Note")
           "* %?\n:PROPERTIES:\n:CREATED: %U\n:END:\n"
            :empty-lines 1)
          ("i" "Ideas-Todo" entry (file+headline org-agenda-file-note "Idea")
           "* TODO [#B] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:END:\n %?"
            :empty-lines 1)
          ("w" "Work" entry (file+headline org-agenda-file-gtd "Work")
           "* TODO [#A] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:END:\n %?"
            :empty-lines 1)
          ("c" "Chrome" entry (file+headline org-agenda-file-note "Quick Note")
           "* TODO [#C] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:URL: %(retrieve-chrome-current-tab-url)\n:END:\n %?"
            :empty-lines 1)
          ("l" "Links" entry (file+headline org-agenda-file-note "Quick Note")
           "* TODO [#C] %^{Heading}\n:PROPERTIES:\n:CREATED: %U\n:LINK: %a\n:END:\n %?"
            :empty-lines 1)
          ("j" "Journal Entry"
            entry (file+datetree org-agenda-file-journal)
            "* %U - %^{Heading}\n  %?"
            :empty-lines 1)))
  (setq org-agenda-custom-commands
        '(
          ("w" . "Works")
          ("wa" "Important-Urgent" tags-todo "+PRIORITY=\"A\"")
          ("wb" "Important-NotUrgent" tags-todo "-Weekly-Monthly-Daily+PRIORITY=\"B\"")
          ("wc" "NotImport" tags-todo "+PRIORITY=\"C\"")
          ;; ("b" "Blog" tags-todo "BLOG")
          ("p" . "Projects")
          ("pw" tags-todo "PROJECT+WORK+CATEGORY=\"work\"")
          ("pl" tags-todo "PROJECT+DREAM+CATEGORY=\"zilongshanren\"")
          ("W" "Weekly Review"
           ((stuck "") ;; review stuck projects as designated by org-stuck-projects
            (tags-todo "PROJECT") ;; review all projects (assuming you use todo keywords to designate projects)
            ))))
  (setq org-bullets-bullet-list '("🐳" "🐬" "🐠" "🐟"))
  (add-hook 'org-mode-hook 'emojify-mode)
  (add-hook 'org-mode-hook 'auto-fill-mode)
  ;; (add-hook 'org-mode-hook 'org-toggle-pretty-entities)
  (evil-define-key 'normal org-mode-map (kbd "<up>") 'org-shiftup)
  (evil-define-key 'normal org-mode-map (kbd "<down>") 'org-shiftdown)
  (evil-define-key 'normal org-mode-map (kbd "<left>") 'org-shiftleft)
  (evil-define-key 'normal org-mode-map (kbd "<right>") 'org-shiftright)
  (evil-define-key 'normal org-mode-map (kbd "C-c j") 'outline-next-visible-heading)
  (evil-define-key 'normal org-mode-map (kbd "C-c k") 'outline-previous-visible-heading)
  (evil-define-key 'normal org-mode-map (kbd "C-c h") 'outline-backward-same-level)
  (evil-define-key 'normal org-mode-map (kbd "C-c l") 'outline-forward-same-level)
  (setq org-agenda-custom-commands
        '(("f" occur-tree "FIXME")))


  (when (version<= "9.2" (org-version))
    (require 'org-tempo))
  ;; ox-pandoc
  ;; default options for all output formats
  (setq org-pandoc-options '((standalone . t)))
  ;; cancel above settings only for 'docx' format
  (setq org-pandoc-options-for-docx '((standalone . nil)))
  ;; special settings for beamer-pdf and latex-pdf exporters
  (setq org-pandoc-options-for-beamer-pdf '((pdf-engine . "xelatex")))
  (setq org-pandoc-options-for-latex-pdf '((pdf-engine . "pdflatex")))
  ;; special extensions for markdown_github output
  (setq org-pandoc-format-extensions '(markdown_github+pipe_tables+raw_html))
  ;; other
  (setenv "WORKON_HOME" "/home/zxf/anaconda3/envs")
  (add-hook 'prog-mode-hook 'yas-minor-mode)
  (with-eval-after-load 'yasnippet
    (define-key yas-minor-mode-map (kbd "<tab>") nil)
    (define-key yas-minor-mode-map (kbd "TAB") nil)
    (define-key yas-minor-mode-map (kbd "<C-tab>") 'yas-expand)
    )
  (with-eval-after-load 'evil
    ;; myfunc
    (defun zxf/move-to-middle ()
      (interactive)
      (let* ((begin (line-beginning-position))
             (end (line-end-position))
             (middle (/ (+ end begin) 2)))
            (goto-char middle)))
    ;; (define-key evil-normal-state-map (kbd "") 'evil-first-non-blank)
    (define-key evil-normal-state-map (kbd "H") 'evil-first-non-blank)
    (define-key evil-normal-state-map (kbd "M") 'zxf/move-to-middle)
    (define-key evil-normal-state-map (kbd "L") 'evil-end-of-line)
    (define-key evil-normal-state-map (kbd "Q") 'evil-record-macro)

    ;; yas
    (evil-define-key 'insert yas-minor-mode-map (kbd "C-e") 'yas-expand)

    ;; evil
    (setq evil-emacs-state-modes (delq 'ibuffer-mode  evil-emacs-state-modes))
    (setq evil-emacs-state-modes (delq 'proced-mode  evil-emacs-state-modes))
    (setq evil-emacs-state-modes (delq 'spacemacs-buffer-mode  evil-emacs-state-modes))
    (setq evil-emacs-state-modes (delq 'help-mode  evil-emacs-state-modes))
    ;; (setq evil-emacs-state-modes (delq 'Custom-mode  evil-emacs-state-modes))
    (define-key evil-normal-state-map (kbd "C-j") #'flycheck-next-error)
    (define-key evil-normal-state-map (kbd "C-k") #'flycheck-previous-error)
    ;; (define-key evil-normal-state-map (kbd "C-x C-k") #'kill-buffer-and-window)
    (define-key evil-normal-state-map (kbd "q") #'kill-buffer-and-window)
    (evil-define-key 'normal org-capture-mode-map (kbd "q") 'org-capture-kill)
    ;; helm swoop
    (defun helm-swoop-from-evil-search ()
      (interactive)
      (if (string-match "\\(evil.*search*\\)" (symbol-name real-last-command))
          (helm-swoop :$query (if isearch-regexp
                                  isearch-string
                                (regexp-quote isearch-string)))
        (helm-swoop)))
    (define-key evil-normal-state-map (kbd "C-s") 'helm-swoop)
    (define-key evil-motion-state-map (kbd "C-s") 'helm-swoop-from-evil-search) ; but didn't work
    (defun my-ibuffer-list-buffers()
      (interactive)
      (ibuffer-list-buffers)
      (other-window 1)
      )
    (define-key spacemacs-buffer-mode-map (kbd "<SPC>fF") 'helm-projectile-find-file-in-known-projects)
    (define-key spacemacs-buffer-mode-map (kbd "C-b") 'lazy-helm/helm-mini)
    (define-key spacemacs-buffer-mode-map (kbd "C-f") 'helm-projectile-find-file)
    (define-key spacemacs-buffer-mode-map (kbd "C-p") 'helm-projectile-switch-project)
    ;; C-q C-backspace to insert the ^? (not actually question mark)
    (define-key key-translation-map (kbd "C-h") "")
    (define-key key-translation-map (kbd "`") (kbd "C-h"))
    (define-key evil-normal-state-map (kbd "<SPC>fF") 'helm-projectile-find-file-in-known-projects)
    (define-key evil-normal-state-map (kbd "C-b") 'lazy-helm/helm-mini)
    (define-key evil-normal-state-map (kbd "C-f") 'helm-projectile-find-file)
    (define-key evil-normal-state-map (kbd "C-p") 'helm-projectile-switch-project)
    (define-key evil-normal-state-map (kbd "C-e") 'evil-iedit-state/iedit-mode)
    (define-key evil-visual-state-map (kbd "C-e") 'evil-iedit-state/iedit-mode)
    (define-key evil-normal-state-map (kbd "M-j") 'move-text-down)
    (define-key evil-normal-state-map (kbd "M-k") 'move-text-up)
    (define-key evil-normal-state-map (kbd "C-x t") 'spacemacs/time-machine-transient-state/body)
    (define-key evil-normal-state-map (kbd "C-x C-u") 'spacemacs/helm-jump-in-buffer)
    (define-key evil-normal-state-map (kbd "C-x C-l") 'spacemacs/comment-or-uncomment-lines)
    (define-key evil-normal-state-map (kbd "C-x C-j") 'dired-jump)
    (define-key evil-normal-state-map (kbd "<SPC> bl") 'my-ibuffer-list-buffers)
    (define-key evil-normal-state-map (kbd "/") 'spacemacs/helm-project-smart-do-search)
    (define-key evil-normal-state-map (kbd "<SPC> /") 'spacemacs/helm-files-smart-do-search)
    ;; parrot
    (define-key evil-normal-state-map (kbd "[r") 'parrot-rotate-prev-word-at-point)
    (define-key evil-normal-state-map (kbd "]r") 'parrot-rotate-next-word-at-point)
    (evil-define-key 'normal dired-mode-map (kbd "gg") 'evil-goto-first-line)
    (evil-define-key 'normal dired-mode-map (kbd "G") 'evil-goto-line)
    (evil-define-key 'normal dired-mode-map (kbd "C-x g") 'magit-status)
    (evil-define-key 'normal helm-swoop-edit-map (kbd "C-c C-c") 'helm-swoop--edit-complete)
    ;; docker contianer
    (evil-define-key 'normal docker-container-mode-map (kbd "a") 'docker-container-attach-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "b") 'docker-container-shell-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "C") 'docker-container-cp-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "d") 'docker-container-diff-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "D") 'docker-container-rm-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "f") 'docker-container-find-file-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "I") 'docker-container-inspect-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "K") 'docker-container-kill-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "l") 'docker-container-ls-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "L") 'docker-container-logs-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "m") 'tablist-mark-forward)
    (evil-define-key 'normal docker-container-mode-map (kbd "o") 'docker-container-stop-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "p") 'docker-container-pause-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "q") 'tablist-quit)
    (evil-define-key 'normal docker-container-mode-map (kbd "r") 'docker-container-restart-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "R") 'docker-container-rename-selection)
    (evil-define-key 'normal docker-container-mode-map (kbd "s") 'docker-container-start-popup)
    (evil-define-key 'normal docker-container-mode-map (kbd "u") 'tablist-unmark-forward)
    (evil-define-key 'normal docker-container-mode-map (kbd "?") 'docker-container-help-popup)
    ;; docker image
    (evil-define-key 'normal docker-image-mode-map (kbd "D") 'docker-image-rm-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "F") 'docker-image-pull-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "I") 'docker-image-inspect-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "l") 'docker-image-ls-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "m") 'tablist-mark-forward)
    (evil-define-key 'normal docker-image-mode-map (kbd "P") 'docker-image-push-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "q") 'tablist-quit)
    (evil-define-key 'normal docker-image-mode-map (kbd "R") 'docker-image-run-popup)
    (evil-define-key 'normal docker-image-mode-map (kbd "T") 'docker-image-tag-selection)
    (evil-define-key 'normal docker-image-mode-map (kbd "u") 'tablist-unmark-forward)
    (evil-define-key 'normal docker-image-mode-map (kbd "?") 'docker-image-help-popup)
    ;; vim like
    (progn
      (spacemacs|define-transient-state my-evil-numbers
        :title "Evil Numbers Transient State, powered by zxf"
        :doc
        "\n[_C-a_] increase number  [_C-x_] decrease  [0..9] prefix  [_q_] quit"
        :bindings
        ("C-a" evil-numbers/inc-at-pt)
        ("C-x" evil-numbers/dec-at-pt)
        ("q" nil :exit t))
      (evil-define-key 'normal global-map (kbd "C-a") 'spacemacs/my-evil-numbers-transient-state/evil-numbers/inc-at-pt)
      (evil-define-key 'normal global-map (kbd "C-x C-x") 'spacemacs/my-evil-numbers-transient-state/evil-numbers/dec-at-pt)
      )
    ;; my func
    (defun my-evil-ctrl-u ()
      (interactive)
      (if (looking-back "^" 0)
          (backward-delete-char 1)
        (if (looking-back "^\s*" 0)
            (delete-region (point) (line-beginning-position))
          (evil-delete (+ (line-beginning-position) (current-indentation)) (point)))))
    ;; underscore word
    (add-hook 'python-mode-hook #'(lambda () (modify-syntax-entry ?_ "w")))
    ;; simulate c-u in vim insert mode behavior
    (define-key evil-insert-state-map (kbd "C-u") 'my-evil-ctrl-u)
    ;; vim-defalut window move
    (define-key evil-motion-state-map (kbd "C-w C-j") #'evil-window-down)
    (define-key evil-motion-state-map (kbd "C-w C-k") #'evil-window-up)
    (define-key evil-motion-state-map (kbd "C-w C-h") #'evil-window-left)
    (define-key evil-motion-state-map (kbd "C-w <DEL>") #'evil-window-left)
    (define-key evil-motion-state-map (kbd "C-w C-l") #'evil-window-right)
    (define-key evil-motion-state-map (kbd "C-w C-w") #'evil-window-next)
    )
  (with-eval-after-load 'helm-swoop
    (define-key helm-swoop-map (kbd "C-s") 'helm-multi-swoop-all-from-helm-swoop)
    )
  (with-eval-after-load 'company
    (define-key company-active-map (kbd "C-w") 'evil-delete-backward-word)
    )
  (with-eval-after-load 'helm
    ;; C-w
    (define-key helm-map (kbd "C-w") 'evil-delete-backward-word)
    (define-key helm-find-files-map (kbd "C-w") 'evil-delete-backward-word)
    ;; C-u
    (define-key helm-map (kbd "C-u") 'my-evil-ctrl-u)
    (define-key helm-find-files-map (kbd "C-u") 'my-evil-ctrl-u)
    )
  (with-eval-after-load 'flycheck-error-list
    (define-key flycheck-error-list-mode-map (kbd "C-w C-j") #'evil-window-down)
    (define-key flycheck-error-list-mode-map (kbd "C-w C-k") #'evil-window-up)
    (define-key flycheck-error-list-mode-map (kbd "C-w <DEL>") #'evil-window-left)
    (define-key flycheck-error-list-mode-map (kbd "C-w C-l") #'evil-window-right)
    (define-key flycheck-error-list-mode-map (kbd "C-w C-w") #'evil-window-next)
    )

  ;; DEBUG
  (defun spacemacs/python-toggle-breakpoint ()
    "Add a break point, highlight it."
    (interactive)
    (let ((trace (cond ((spacemacs/pyenv-executable-find "trepan3k") "import trepan.api; trepan.api.debug()")
                       ((spacemacs/pyenv-executable-find "wdb") "__import__('wdb').set_trace()")
                      ((spacemacs/pyenv-executable-find "ipdb") "__import__('ipdb').set_trace()")
                      ((spacemacs/pyenv-executable-find "pudb") "__import__('pudb').set_trace()")
                      ((spacemacs/pyenv-executable-find "ipdb3") "__import__('ipdb').set_trace()")
                      ((spacemacs/pyenv-executable-find "pudb3") "__import__('pudb').set_trace()")
                      ((spacemacs/pyenv-executable-find "python3.7") "breakpoint()")
                      ((spacemacs/pyenv-executable-find "python3.8") "breakpoint()")
                      (t "__import__('pdb').set_trace()")))
          (line (thing-at-point 'line)))
      (if (and line (string-match trace line))
          (kill-whole-line)
        (progn
          (back-to-indentation)
          (insert trace)
          (insert "\n")
          (python-indent-line)))))

  ;; (setq python-shell-interpreter "python"
  ;;       python-shell-interpreter-args "-m IPython --simple-prompt -i")
  ;; (with-eval-after-load 'python
  ;;   (add-hook 'python-mode-hook (lambda () (setq python-shell-interpreter "python"))))
  (evil-define-key 'normal python-mode-map (kbd "C-c r") 'pyvenv-restart-python)
  (evil-define-key 'normal python-mode-map (kbd "C-c w") 'pyvenv-workon)
  (evil-define-key 'normal python-mode-map (kbd "C-c d") 'spacemacs/python-toggle-breakpoint)
  (evil-define-key 'normal python-mode-map (kbd "C-c I") 'py-isort-buffer)
  (evil-define-key 'normal python-mode-map (kbd "C-c i") 'spacemacs/python-remove-unused-imports)
  (require 'py-autopep8)
  (add-hook 'python-mode-hook 'py-autopep8-enable-on-save)
  ;; (add-hook 'before-save-hook 'py-isort-before-save)
  (add-hook 'c++-mode-hook (lambda () (setq flycheck-clang-language-standard "c++11")))
  ;; proced
  (evil-define-key 'normal proced-mode-map (kbd "d") 'proced-mark)
  (evil-define-key 'normal proced-mode-map (kbd "m") 'proced-mark)
  (evil-define-key 'normal proced-mode-map (kbd "x") 'proced-send-signal)
  (evil-define-key 'normal proced-mode-map (kbd "u") 'proced-unmark)
  (evil-define-key 'normal proced-mode-map (kbd "q") 'quit-window)
  )
;; Do not write anything past this comment. This is where Emacs will
;; auto-generate custom variable definitions.
(setq custom-file (expand-file-name "custom.el" dotspacemacs-directory))
(load custom-file 'no-error 'no-message)
