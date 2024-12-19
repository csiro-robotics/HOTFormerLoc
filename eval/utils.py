def get_query_database_splits(params):
    if params.dataset_name == 'Oxford':
        eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                               'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']
        eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                            'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']
    elif params.dataset_name == 'MulRan':
        eval_database_files = ['DCC_database.pickle', 'Sejong_database.pickle']
        eval_query_files = ['DCC_queries.pickle', 'Sejong_queries.pickle']
    elif 'CSWildPlaces' in params.dataset_name:
        eval_database_files = [
            'CSWildPlaces_Karawatha_evaluation_database.pickle',
            'CSWildPlaces_Venman_evaluation_database.pickle',
            'CSWildPlaces_QCAT_evaluation_database.pickle', 
            'CSWildPlaces_Samford_evaluation_database.pickle',
        ]
        eval_query_files = [
            'CSWildPlaces_Karawatha_evaluation_query.pickle',
            'CSWildPlaces_Venman_evaluation_query.pickle',
            'CSWildPlaces_QCAT_evaluation_query.pickle', 
            'CSWildPlaces_Samford_evaluation_query.pickle',
        ]
    elif 'WildPlaces' in params.dataset_name:
        eval_database_files = [
            'Karawatha_evaluation_database.pickle',
            'Venman_evaluation_database.pickle',
        ]
        eval_query_files = [
            'Karawatha_evaluation_query.pickle',
            'Venman_evaluation_query.pickle',            
        ]
    elif params.dataset_name == 'CSCampus3D':
        eval_database_files = ['umd_evaluation_database.pickle']
        eval_query_files = ['umd_evaluation_query_v2.pickle']
    else:
        raise NotImplementedError(f'Dataset {params.dataset_name} has no splits implemented')
    return eval_database_files, eval_query_files