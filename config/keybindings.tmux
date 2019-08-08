# keybindings
unbind-key C-b
# set-option -g prefix C-b
set-option -g prefix C-n
bind-key C-n last-window
bind-key x kill-window
bind-key p paste-buffer
bind r source-file /usr/share/byobu/profiles/tmuxrc \; display-message "LONG LIVE ZXF!!!"
bind-key C-d detach-client
bind-key p paste-buffer
bind-key v split-window -h


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
set-window-option -g automatic-rename
set -g status-interval 4
set -g status-left '#[bg=colour247]#[fg=colour16]#{?client_prefix,#[bg=colour124]#[fg=colour249],} ♂ #S #[bg=default]#[fg=colour247]#{?client_prefix,#[fg=colour124],}#{?window_zoomed_flag, ,}#[fg=colour40]#[bg=default] ♥ #(~/dotfiles/tmux_scripts/battery.sh)'
set -g window-status-current-format "#[bg=colour99]#[fg=colour0]#[bg=colour99]#[fg=colour250] ☻#I #[bg=colour99]#[fg=colour250]#W #[fg=colour99]#[bg=default]"
set -g window-status-format "#[fg=colour146]☹#[fg=colour146]#I #[fg=146]#W #[fg=colour99]"