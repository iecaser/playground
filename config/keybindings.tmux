# keybindings
unbind-key C-b
# set-option -g prefix C-b
# bind-key C-b last-window
set-option -g prefix C-n
bind-key C-n last-window
bind-key x kill-window
# bind-key C-b last-window
set -g status-position bottom
bind-key C-d detach-client
bind-key p paste-buffer
# bind-key -T copy-mode-vi v send-keys -X begin-selection
# bind-key -T copy-mode-vi y send-keys -X copy-selection-and-cancel
# reload config
bind r source-file /usr/share/byobu/profiles/tmuxrc \; display-message "Config reloaded..."

set -g status-position bottom
set -s escape-time 0
set-window-option -g mode-keys vi
# vi mode
# "set-window-option -g mode-keys vi" works in tmux 2.1 and above.
# "setw -g mode-keys vi" works in tmux 1.8
set -g status-keys vi
# color
set -g default-terminal "xterm-256color"

# set window split
bind-key v split-window -h
