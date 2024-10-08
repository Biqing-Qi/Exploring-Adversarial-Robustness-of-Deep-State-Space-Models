from models.SSM import SSM, Mega, S5_SSM, S6_SSM

def build_model(args, model_name):
    if model_name == 'SSM':
        if args.use_AdSS:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, \
                use_AdSS=True,AdSS_Type=args.AdSS_Type)
        else:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers)
    elif model_name == 'DSS':
        if args.use_AdSS:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, mode = 'diag', \
                use_AdSS=True,AdSS_Type=args.AdSS_Type)
        else:
            model = SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers, mode = 'diag')
    elif model_name == 'S5':
        model = S5_SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers)
    elif model_name == 'Mega':
        model = Mega(d_input=1, d_model=128, hidden_dim=32, n_layers=args.num_layers, seq_len=28*28) 
    elif model_name == 'S6':
        model = S6_SSM(d_input=1, d_model=128, d_state=32, n_layers=args.num_layers)     
    return model