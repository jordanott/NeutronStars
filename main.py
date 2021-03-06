import neutron_stars as ns


args = ns.parse_args()

ns.paradigm_settings(args)

data_loader = ns.DataLoader(args)

if args['run_type'] == 'train':
    model = ns.build_model(args)

elif args['run_type'] == 'test':
    pass
