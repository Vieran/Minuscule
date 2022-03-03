# 修复Arch Linux

*前情提要：尝试LFS，然后原本系统的grub写成了LFS的，而LFS没安装好，导致原系统无法启动。进入了grub的正常模式*

```bash
# 因为arch是图形界面开启的，所以直接搞不行，要设置图形界面的grub，按照“参考”的第一个链接设置（这个时候只能暂时用某个文件，因为没法复制移动）
# 进入security，挂载
mount -t ext4 /dev/sda3 /root
# 在/boot/grub/grub.cfg按照“参考”第一个链接在文件内设置相关参数，让它下次启动的时候可以直接进入图形模式
# 重启

# 这时候看到grub和之前不同了，这是可以进入图形界面的
# 正常设置启动
root=(hd0, gpt3)
linux /boot/vm... root=/dev/sda3
initrd /boot/initrd-linux.img
boot

# 修复grub文件
su
grub-mkconfig -o /boot/grub/gurb.cfg

# 系统恢复正常
```



## 参考

[GRUB- "No suitable mode found" - 仙贝 - 博客园 (cnblogs.com)](https://www.cnblogs.com/lymi/p/4894589.html)

[「GRUB」- 在BIOS系统上的GRUB引导 - K4NZ BLOG](https://blog.k4nz.com/c81512543fe97e679c57e09156d0ab72/)

[「Grub」- 手动引导启动 - K4NZ BLOG](https://blog.k4nz.com/a49f9c3be9e0e8ba76e7d702bb21f7c6/)

[开机进入GRUB不要慌，命令行也可启动Linux - zztian - 博客园 (cnblogs.com)](https://www.cnblogs.com/zztian/p/10289083.html)