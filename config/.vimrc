set smartindent
set autoindent
set tabstop=4
set shiftwidth=4
set softtabstop=4
set smarttab
set expandtab
set shiftround
set nrformats=
set encoding=utf-8
autocmd! bufwritepost .vimrc source %
nnoremap k gk
nnoremap gk k
nnoremap j gj
nnoremap gj j
inoremap kj <Esc>
nnoremap <leader>q :q<CR>
nnoremap <leader>w :w<CR>
set background=dark
set t_Co=256
let mapleader = "\<Space>"
let g:mapleader = "\<Space>"
set history=2000
filetype on
filetype indent on
set cursorcolumn
set cursorline 