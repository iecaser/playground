(+ 2 2)
(+ 2 (+ 1 1))
(setq my-name "zxf")
(message my-name)
(defun my-func ()
  (interactive)
  (message my-name))
(my-func)
(global-set-key (kbd "<f4>") 'my-func)
