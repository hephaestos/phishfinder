#! /usr/bin/awk -f
!NF{
    if(blank_lines++ > 1){
        print
    }
}
/^From: /{
    print
    next
}
{
    if(blank_lines > 1) {
        print
    }
}
