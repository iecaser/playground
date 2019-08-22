# keybindings
unbind-key C-b
# set-option -g prefix C-b
set-option -g prefix C-n
bind-key C-n last-window
bind-key x kill-window
bind-key p paste-buffer
bind r source-file /usr/share/byobu/profiles/tmuxrc \; display-message "LONG LIVE ZXF!!!"
bind-key C-d detach-client
# v2.3 or below
# bind-key -t vi-copy 'v' begin-selection
# bind-key -t vi-copy 'y' copy-selection
bind-key -T copy-mode-vi v send-keys -X begin-selection
bind-key -T copy-mode-vi y send-keys -X copy-selection \; send-keys -X cancel
bind-key p paste-buffer

# Not konwn
set -g status-keys vi
set-window-option -g mode-keys vi
setw -g aggressive-resize on
set -s escape-time 0

# UI
## common
set -g default-terminal "xterm-256color"
## status bar
set -g base-index 1
set-option -g allow-rename
set -g status-interval 4
set -g status-left '#[bg=colour247]#[fg=colour16]#{?client_prefix,#[bg=colour24]#[fg=colour249],} ♂ #S #[bg=default]#[fg=colour247]#{?client_prefix,#[fg=colour24],}#{?window_zoomed_flag, ,}#[fg=colour1]#[bg=default] ♥ #(~/dotfiles/tmux_scripts/battery.sh)'
set -g window-status-current-format "#[bg=colour99]#[fg=colour0]#[bg=colour99]#[fg=colour250] ☻#I #[bg=colour99]#[fg=colour250]#W #[fg=colour99]#[bg=default]"
set -g window-status-format "#[fg=colour146]☹#[fg=colour146]#I #[fg=146]#W #[fg=colour99]"
