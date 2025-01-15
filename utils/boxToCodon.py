def box_to_codon(input_str):
    """
    输入氨基酸密码子对形式的字符串，返回密码子形式的字符串
    另外还返回了输入字符串长度输出字符串长度以及出现错误的次数用于校验
    """
    lines = input_str.split('\n')
    output_str=''
    codon_seq = ''
    count = 0 
    err_count = 0
    seq_len = 0
    for line in lines:
        line = line.strip()
        if line == '':
            output_str += codon_seq + '\n'
            seq_len += len(codon_seq)
            codon_seq = ''
        elif line[0] == '#':
            continue
        else:
            count += 1
            if(line == 'F a'):
                codon_seq = codon_seq + 'TTT'
            elif(line == 'F b'):
                codon_seq = codon_seq + 'TTC'
            elif(line == 'L c'):
                codon_seq = codon_seq + 'TTA'
            elif(line == 'L d'):
                codon_seq = codon_seq + 'TTG'
            elif(line == 'S b'):
                codon_seq = codon_seq + 'TCT'
            elif(line == 'S f'):
                codon_seq = codon_seq + 'TCC'
            elif(line == 'S h'):
                codon_seq = codon_seq + 'TCA'
            elif(line == 'S g'):
                codon_seq = codon_seq + 'TCG'
            elif(line == 'Y c'):
                codon_seq = codon_seq + 'TAT'
            elif(line == 'Y h'):
                codon_seq = codon_seq + 'TAC'
            elif(line == 'X w'):
                codon_seq = codon_seq + 'TAA'
            elif(line == 'C d'):
                codon_seq = codon_seq + 'TGT' 
            elif(line == 'C g'):
                codon_seq = codon_seq + 'TGC'
            elif(line == 'W k'):
                codon_seq = codon_seq + 'TGG'     
            elif(line == 'L b'):
                codon_seq = codon_seq + 'CTT'
            elif(line == 'L f'):
                codon_seq = codon_seq + 'CTC'
            elif(line == 'L h'):
                codon_seq = codon_seq + 'CTA'
            elif(line == 'L g'):
                codon_seq = codon_seq + 'CTG'
            elif(line == 'P f'):
                codon_seq = codon_seq + 'CCT'
            elif(line == 'P l'):
                codon_seq = codon_seq + 'CCC'
            elif(line == 'P m'):
                codon_seq = codon_seq + 'CCA'
            elif(line == 'P n'):
                codon_seq = codon_seq + 'CCG'
            elif(line == 'H h'):
                codon_seq = codon_seq + 'CAT'
            elif(line == 'H m'):
                codon_seq = codon_seq + 'CAC'
            elif(line == 'Q o'):
                codon_seq = codon_seq + 'CAA'
            elif(line == 'Q r'):
                codon_seq = codon_seq + 'CAG'
            elif(line == 'R g'):
                codon_seq = codon_seq + 'CGT'
            elif(line == 'R n'):
                codon_seq = codon_seq + 'CGC'
            elif(line == 'R r'):
                codon_seq = codon_seq + 'CGA'
            elif(line == 'R s'):
                codon_seq = codon_seq + 'CGG'
            elif(line == 'I c'):
                codon_seq = codon_seq + 'ATT'
            elif(line == 'I h'):
                codon_seq = codon_seq + 'ATC'
            elif(line == 'I i'):
                codon_seq = codon_seq + 'ATA'
            elif(line == 'M j'):
                codon_seq = codon_seq + 'ATG'
            elif(line == 'T h'):
                codon_seq = codon_seq + 'ACT'
            elif(line == 'T m'):
                codon_seq = codon_seq + 'ACC'
            elif(line == 'T o'):
                codon_seq = codon_seq + 'ACA'
            elif(line == 'T r'):
                codon_seq = codon_seq + 'ACG'
            elif(line == 'N i'):
                codon_seq = codon_seq + 'AAT'
            elif(line == 'N o'):
                codon_seq = codon_seq + 'AAC'
            elif(line == 'K u'):
                codon_seq = codon_seq + 'AAG'
            elif(line == 'K t'):
                codon_seq = codon_seq + 'AAA'
            elif(line == 'S j'):
                codon_seq = codon_seq + 'AGT'
            elif(line == 'S r'):
                codon_seq = codon_seq + 'AGC'
            elif(line == 'R u'):
                codon_seq = codon_seq + 'AGA'
            elif(line == 'R q'):
                codon_seq = codon_seq + 'AGG'
            elif(line == 'V d'):
                codon_seq = codon_seq + 'GTT'
            elif(line == 'V g'):
                codon_seq = codon_seq + 'GTC'
            elif(line == 'V j'):
                codon_seq = codon_seq + 'GTA'
            elif(line == 'V k'):
                codon_seq = codon_seq + 'GTG'
            elif(line == 'A g'):
                codon_seq = codon_seq + 'GCT'
            elif(line == 'A n'):
                codon_seq = codon_seq + 'GCC'
            elif(line == 'A r'):
                codon_seq = codon_seq + 'GCA'
            elif(line == 'A s'):
                codon_seq = codon_seq + 'GCG'
            elif(line == 'D j'):
                codon_seq = codon_seq + 'GAT'
            elif(line == 'D r'):
                codon_seq = codon_seq + 'GAC'
            elif(line == 'E u'):
                codon_seq = codon_seq + 'GAA'
            elif(line == 'E q'):
                codon_seq = codon_seq + 'GAG'
            elif(line == 'G k'):
                codon_seq = codon_seq + 'GGT'
            elif(line == 'G s'):
                codon_seq = codon_seq + 'GGC'
            elif(line == 'G q'):
                codon_seq = codon_seq + 'GGA'
            elif(line == 'G p'):
                codon_seq = codon_seq + 'GGG'
            # 有错误的部分
            elif(line == 'N w'):
                codon_seq = codon_seq + 'AAC'
            elif(line == 'E w'):
                codon_seq = codon_seq + 'GAA'
            else:
                # print("不一致的："+ line+"\n")
                err_count += 1
    
    return count, err_count, seq_len,output_str
    