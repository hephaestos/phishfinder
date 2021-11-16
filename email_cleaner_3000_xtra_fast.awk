#! /usr/bin/awk -f
!NF {blank_lines++}
END {
    print blank_lines
}

